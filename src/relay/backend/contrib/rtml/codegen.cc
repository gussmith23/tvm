#include <tvm/runtime/registry.h>

#include <numeric>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

class CodegenRTML : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenRTML(const std::string& id) { this->ext_func_id_ = id; }
  std::string JIT(const std::vector<Output>& out) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;
    CHECK(op_name == "nn.dense");
    CHECK(call->args.size() == 2);
    auto res = VisitExpr((call->args[0]));
    CHECK(res.size() == 1);
    const auto activations_output = res[0];
    res = VisitExpr((call->args[1]));
    CHECK(res.size() == 1);
    const auto weights_output = res[0];

    const auto activations_type = call->args[0]->checked_type().as<TensorTypeNode>();
    CHECK(activations_type);
    CHECK(activations_type->shape.size() == 2);
    const auto batch_size = activations_type->shape[0].as<IntImmNode>()->value;
    const auto input_vector_size = activations_type->shape[1].as<IntImmNode>()->value;

    const auto weights_type = call->args[1]->checked_type().as<TensorTypeNode>();
    CHECK(weights_type);
    CHECK(weights_type->shape.size() == 2);
    CHECK(weights_type->shape[1].as<IntImmNode>()->value == input_vector_size);
    const auto output_vector_size = weights_type->shape[0].as<IntImmNode>()->value;

    Output output;
    const std::string out = "buf_" + std::to_string(buf_idx_++);
    const auto out_size = GetShape1DSize(call->checked_type());
    output.name = out;
    output.size = out_size;
    output.dtype = GetDtypeString(call->checked_type().as<TensorTypeNode>());
    output.need_copy = true;
    buf_decl_.push_back("float* " + out + " = (float*)std::malloc(4 * " + std::to_string(out_size) +
                        ");");

    std::ostringstream rtml_call;
    rtml_call << "rtml_systolic_array_weight_stationary_fc("  //
              << "0,"                                         // Hardware ID
              << out << ","                                   //
              << activations_output.name << ", "              //
              << weights_output.name << ", "                  //
              << input_vector_size << ","                     //
              << output_vector_size << ","                    //
              << batch_size << ");";                          //

    ext_func_body_.push_back(rtml_call.str());

    return {output};
  }

 private:
  struct GenerateBodyOutput {
    std::string decl;
    std::vector<std::string> buffers;
    std::vector<Output> outputs;
  };

  size_t buf_idx_{0};
  std::string ext_func_id_{""};
  Array<String> const_vars_{};
  Array<Var> ext_func_args_{};
  std::vector<std::string> ext_func_body_{};
  std::vector<std::string> buf_decl_{};
  std::string const_array_name_{""};

  friend class RTMLModuleCodegen;
};

class RTMLModuleCodegen : public CSourceModuleCodegenBase {
 public:
  std::pair<std::string, Array<String>> GenRTMLFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenRTML builder(sid);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);

    return {sid, builder.const_vars_};
  }

  runtime::Module CreateCSourceModule(const ObjectRef& ref) {
    // Create headers
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    // code_stream_ << "#include <tvm/runtime/container.h>\n";
    // code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    // code_stream_ << "using namespace tvm::runtime;\n";
    // code_stream_ << "using namespace tvm::runtime::contrib;\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_
        << "extern \"C\" void "
           "rtml_systolic_array_weight_stationary_fc(int,float*,float*,float*,int,int,int);\n";
    code_stream_ << "\n";

    CHECK(ref->IsInstance<FunctionNode>());
    auto res = GenRTMLFunc(Downcast<Function>(ref));
    std::string code = code_stream_.str();
    String sym = std::get<0>(res);
    Array<String> variables = std::get<1>(res);

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", Array<String>{sym}, variables);
  }

 private:
  std::ostringstream code_stream_;
};

runtime::Module RTMLCompiler(const ObjectRef& ref) {
  RTMLModuleCodegen rtml;
  return rtml.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.rtml").set_body_typed(RTMLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm