#include <tvm/runtime/registry.h>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class CodegenRTML : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  std::string JIT(const std::vector<Output>& out) { return ""; }
  runtime::Module CreateCSourceModule(const ObjectRef& ref) { return runtime::Module(); }
};

runtime::Module RTMLCompiler(const ObjectRef& ref) {
  CodegenRTML rtml;
  return rtml.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.rtml").set_body_typed(RTMLCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm