# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_POSIT)
  message(STATUS "Build with contrib.posit")
  if (NOT UNIVERSAL_PATH)
    message(FATAL_ERROR "Fail to get Universal path")
  endif(NOT UNIVERSAL_PATH)
  
  include_directories(${UNIVERSAL_PATH}/include)
  list(APPEND RUNTIME_SRCS 3rdparty/posit/posit-wrapper.cc)
endif(USE_POSIT)
