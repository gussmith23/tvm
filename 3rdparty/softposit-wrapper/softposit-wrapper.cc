#include "softposit.h"
#include <iostream>

posit32_t Uint32ToSoftPosit(uint32_t in) {
  return ui32_to_p32(in); //convertFloatToP32(static_cast<float>(in));
}

 uint32_t SoftPositToUint32(posit32_t in) {
  return static_cast<uint32_t>(p32_to_ui32(in));
}

 extern "C" uint32_t FloatToSoftPosit32es2(float in) {
  return 0;
}

 extern "C" float SoftPosit32es2ToFloat(uint32_t in) {
  return 0;
}

 extern "C" uint32_t SoftPosit32es2Add(uint32_t a, uint32_t b) {
  return 0;
}
