#include <tvm/ir.h>
#include <tvm/ir_operator.h>

namespace tvm {

/*!
 * \brief Lowering function used by LowerDatatypes pass when lowering myfloat
 */
TVM_REGISTER_GLOBAL("tvm.datatypes.lower.llvm.cast.myfloat.float")
    .set_body([](runtime::TVMArgs args, runtime::TVMRetValue* rv) {
      Expr e = args[0];
      const ir::Cast* cast = e.as<ir::Cast>();
      internal_assert(cast);
      auto type = cast->type;

      // myfloats are simply the bits of a float stored in a uint.
      *rv = reinterpret(tvm::UInt(type.bits(), type.lanes()), cast->value);
    });

}  // namespace tvm
