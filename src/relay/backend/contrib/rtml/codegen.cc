#include <tvm/runtime/registry.h>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class CodegenRTML : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenRTML(const std::string& id) { this->ext_func_id_ = id; }
  std::string JIT(const std::vector<Output>& out) { return ""; }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    GenerateBodyOutput ret;
    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;
    CHECK(op_name == "nn.dense");

    // buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
    // ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

 private:
  struct GenerateBodyOutput {
    std::string decl;
    std::vector<std::string> buffers;
    std::vector<Output> outputs;
  };

  std::string ext_func_id_;
  Array<String> const_vars_;

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
    // code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    // code_stream_ << "#include <tvm/runtime/container.h>\n";
    // code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    // code_stream_ << "#include <dlpack/dlpack.h>\n";
    // code_stream_ << "using namespace tvm::runtime;\n";
    // code_stream_ << "using namespace tvm::runtime::contrib;\n";
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