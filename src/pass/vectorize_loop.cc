/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vectorize_loop.cc
 */
// Loop vectorizer as in Halide pipeline.
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/arithmetic.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"
#include "../codegen/datatype/registry.h"

namespace tvm {
namespace ir {

inline Expr BroadcastTo(Expr e, int lanes) {
  if (e.dtype().lanes() == lanes) return e;
  if (const Broadcast* op = e.as<Broadcast>()) {
    if (lanes % op->lanes == 0) {
      return Broadcast::make(op->value, lanes);
    }
  }
  CHECK_EQ(e.dtype().lanes(), 1)
      << "Cannot broadcast lane=" << e.dtype().lanes()
      << " to " << lanes;
  return Broadcast::make(e, lanes);
}

// Rewrite vectorized allocation access
// This is necessary for making each vector component containing its own workspace.
// Originates from Halide's loop vectorizer
//
// s[i] = s[i * lanes + var]
//
// The same principle applies when using one thread to simulate multiple context.
//
class VecAllocAccess : public StmtExprMutator {
 public:
  VecAllocAccess(const Variable* buf, Var var, int var_lanes)
      : buf_(buf), var_(var), var_lanes_(var_lanes) {}
  // Load
  Expr VisitExpr_(const Load* op) final {
    Expr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<Load>();
    if (op->buffer_var.get() == buf_) {
      return Load::make(op->dtype, op->buffer_var,
                        op->index * var_lanes_ + var_,
                        op->predicate);
    } else {
      return expr;
    }
  }
  // Store
  Stmt VisitStmt_(const Store* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<Store>();
    if (op->buffer_var.get() == buf_) {
      return Store::make(op->buffer_var,
                         op->value,
                         op->index * var_lanes_ + var_,
                         op->predicate);
    } else {
      return stmt;
    }
  }

 private:
  // buffer var
  const Variable* buf_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
};

class Vectorizer : public StmtExprMutator {
 public:
  Vectorizer(Var var, int var_lanes)
      : var_(var), var_lanes_(var_lanes) {
    ramp_ = Ramp::make(0, 1, var_lanes);
  }

  // Don't run on custom datatypes.
  // TODO(gus) can this be removed now? I think we implemented another way to
  // disable vectorization.
  Expr VisitExpr(Expr expr) final {
    if (tvm::datatype::Registry::Global()->GetTypeRegistered(expr.type().code()))
      return expr;
    return IRMutator::Mutate(expr);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    CHECK(!need_scalarize_);
    Stmt ret = StmtExprMutator::VisitStmt(stmt);
    if (need_scalarize_) {
      need_scalarize_ = false;
      return Scalarize(stmt);
    } else {
      return ret;
    }
  }

  Expr VisitExpr_(const Add* op) final {
    return AddSubVec(op);
  }
  Expr VisitExpr_(const Sub* op) final {
    return AddSubVec(op);
  }
  Expr VisitExpr_(const Mul* op) final {
    Expr a = this->VisitExpr(op->a);
    Expr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return GetRef<Expr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const Ramp* b_ramp = b.as<Ramp>();
        const Ramp* a_ramp = a.as<Ramp>();
        if (a_ramp && b.dtype().lanes() == 1 && analyzer_.CanProve(b > 0)) {
          return Ramp::make(
              a_ramp->base * b, a_ramp->stride * b, a_ramp->lanes);
        }
        if (b_ramp && a.dtype().lanes() == 1 && analyzer_.CanProve(a > 0)) {
          return Ramp::make(
              b_ramp->base * a, b_ramp->stride * a, b_ramp->lanes);
        }
      }
      return Mul::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
    return BinaryVec(op);
  }
  Expr VisitExpr_(const Div* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const Mod* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const FloorDiv* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const FloorMod* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const Min* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const Max* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const EQ* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const NE* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const LT* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const LE* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const GT* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const GE* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const And* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const Or* op) final {
    return BinaryVec(op);
  }
  Expr VisitExpr_(const Ramp* op) final {
    Expr base = this->VisitExpr(op->base);
    Expr stride = this->VisitExpr(op->stride);
    if (base.dtype().lanes() > 1 && stride.dtype().lanes() == 1) {
      const Ramp* base_ramp = base.as<Ramp>();
      if (analyzer_.CanProve(base_ramp->stride == stride * make_const(stride.dtype(), op->lanes))) {
        return Ramp::make(base_ramp->base, stride, op->lanes * base_ramp->lanes);
      }
    }
    int lanes = std::max(base.dtype().lanes(), stride.dtype().lanes());
    base = BroadcastTo(base, lanes);
    stride = BroadcastTo(stride, lanes);
    Array<Expr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp::make(Shuffle::make_extract_element(base, i),
                     Shuffle::make_extract_element(stride, i),
                     op->lanes));
    }
    return Shuffle::make_concat(elems);
  }
  Expr VisitExpr_(const Select *op) final {
    Expr cond = this->VisitExpr(op->condition);
    Expr t = this->VisitExpr(op->true_value);
    Expr f = this->VisitExpr(op->false_value);
    if (cond.same_as(op->condition) &&
        t.same_as(op->true_value) &&
        f.same_as(op->false_value)) {
      return GetRef<Expr>(op);
    } else {
      int lanes = std::max(std::max(
          cond.dtype().lanes(),
          t.dtype().lanes()), f.dtype().lanes());
      return Select::make(cond, BroadcastTo(t, lanes), BroadcastTo(f, lanes));
    }
  }
  Expr VisitExpr_(const Cast *op) final {
    Expr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<Expr>(op);
    } else {
      return Cast::make(op->dtype.with_lanes(value.dtype().lanes()), value);
    }
  }
  // Variable
  Expr VisitExpr_(const Variable* v) final {
    if (v == var_.get()) {
      return ramp_;
    } else if (lets_.count(v)) {
        return lets_[v];
    } else {
      return GetRef<Expr>(v);
    }
  }
  // IfThenElse expr
  Expr MutateIfThenElseExpr_(const Call *op) {
    Expr cond = this->VisitExpr(op->args[0]);
    if (cond.dtype().is_vector())  {
      need_scalarize_ = true;
      return GetRef<Expr>(op);
    }
    Expr t = this->VisitExpr(op->args[1]);
    Expr f = this->VisitExpr(op->args[2]);
    if (cond.same_as(op->args[0]) &&
        t.same_as(op->args[1]) &&
        f.same_as(op->args[2])) {
      return GetRef<Expr>(op);
    } else {
      int lanes = std::max(t.dtype().lanes(), f.dtype().lanes());
      t = BroadcastTo(t, lanes);
      f = BroadcastTo(f, lanes);
      return Call::make(
          op->dtype.with_lanes(lanes), op->name,
          {cond, t, f}, op->call_type, op->func, op->value_index);
    }
  }
  // Call
  Expr VisitExpr_(const Call* op) final {
    if (op->name == intrinsic::tvm_if_then_else) {
      return MutateIfThenElseExpr_(op);
    }
    if (!op->is_vectorizable()) {
      // Cannot vectorize this op
      Array<Expr> new_args;
      for (auto arg : op->args) {
        auto new_arg = this->VisitExpr(arg);
        if (new_arg.dtype().is_vector()) {
          need_scalarize_ = true;
          return GetRef<Expr>(op);
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return GetRef<Expr>(op);
      } else {
        return Call::make(
            op->dtype, op->name, new_args, op->call_type, op->func, op->value_index);
      }
    } else {
      int lane = 0;
      Array<Expr> new_args = MutateArray(op->args, &lane);
      // normal code path.
      if (op->args.same_as(new_args)) {
        return GetRef<Expr>(op);
      } else {
        return Call::make(
            op->dtype.with_lanes(lane), op->name, new_args,
            op->call_type, op->func, op->value_index);
      }
    }
  }
  // Load
  Expr VisitExpr_(const Load* op) final {
    Expr index = this->VisitExpr(op->index);
    Expr pred = this->VisitExpr(op->predicate);
    if (index.same_as(op->index) && pred.same_as(op->predicate)) {
      return GetRef<Expr>(op);
    } else {
      int lanes = std::max(index.dtype().lanes(), pred.dtype().lanes());
      return Load::make(
          op->dtype.with_lanes(lanes),
          op->buffer_var,
          BroadcastTo(index, lanes),
          BroadcastTo(pred, lanes));
    }
  }
  // Let
  Expr VisitExpr_(const Let* op) final {
    Expr value = this->VisitExpr(op->value);
    CHECK(!lets_.count(op->var.get())) << "not SSA";
    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var v(op->var->name_hint, value.dtype());
      lets_[op->var.get()] = v;
      return Let::make(v, value, this->VisitExpr(op->body));
    } else {
      Expr body = this->VisitExpr(op->body);
      if (value.same_as(op->value) &&
          body.same_as(op->body)) {
        return GetRef<Expr>(op);
      } else {
        return Let::make(op->var, value, body);
      }
    }
  }
  // Provide
  Stmt VisitStmt_(const Provide* op) final {
    Expr new_value = this->VisitExpr(op->value);
    int lane = new_value.dtype().lanes();
    Array<Expr> new_args = MutateArray(op->args, &lane);
    if (op->args.same_as(new_args) && op->value.same_as(new_value)) {
      return GetRef<Stmt>(op);
    } else {
      new_value = BroadcastTo(new_value, lane);
      return Provide::make(op->func, op->value_index, new_value, new_args);
    }
  }
  // Store
  Stmt VisitStmt_(const Store* op) final {
    Expr value = this->VisitExpr(op->value);
    Expr index = this->VisitExpr(op->index);
    Expr pred = this->VisitExpr(op->predicate);
    if (value.same_as(op->value) && index.same_as(op->index)) {
      return GetRef<Stmt>(op);
    } else {
      int lanes = std::max(value.dtype().lanes(), index.dtype().lanes());
      lanes = std::max(lanes, pred.dtype().lanes());
      return Store::make(op->buffer_var,
                         BroadcastTo(value, lanes),
                         BroadcastTo(index, lanes),
                         BroadcastTo(pred, lanes));
    }
  }
  // For
  Stmt VisitStmt_(const For* op) final {
    if (op->for_type == ForType::Vectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    CHECK(is_zero(op->min));
    CHECK(!op->extent.dtype().is_vector());
    Expr extent = this->VisitExpr(op->extent);
    if (extent.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt body = this->VisitStmt(op->body);
    if (extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return For::make(
          op->loop_var, op->min, extent,
          op->for_type, op->device_api, body);
    }
  }
  // IfThenElse
  Stmt VisitStmt_(const IfThenElse* op) final {
    CHECK(!op->condition.dtype().is_vector());
    Expr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt then_case = this->VisitStmt(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = this->VisitStmt(op->else_case);
    }
    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return GetRef<Stmt>(op);
    } else {
      return IfThenElse::make(condition, then_case, else_case);
    }
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmt* op) final {
    LOG(WARNING) << "Cannot vectorize with LetStmt, remove it with Simplify Before Vectorize";
    return Scalarize(GetRef<Stmt>(op));
  }
  // Allocate
  Stmt VisitStmt_(const Allocate* op) final {
    if (op->new_expr.defined()) {
      LOG(WARNING) << "Cannot vectorize with new expr";
      return Scalarize(GetRef<Stmt>(op));
    }
    Expr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      LOG(WARNING) << "Cannot handle vector extent in alloc ";
      return Scalarize(GetRef<Stmt>(op));
    }
    Array<Expr> extents;
    for (size_t i = 0; i < op->extents.size(); i++) {
      Expr new_ext = this->VisitExpr(op->extents[i]);
      if (new_ext.dtype().is_vector()) {
        LOG(WARNING) << "Cannot handle vector extent in alloc ";
        return Scalarize(GetRef<Stmt>(op));
      }
      extents.push_back(new_ext);
    }
    // place the vector lanes in least significant dimension.
    extents.push_back(var_lanes_);
    // rewrite access to buffer internally.
    Stmt body = VecAllocAccess(
        op->buffer_var.get(), var_, var_lanes_)(op->body);
    body = this->VisitStmt(body);
    return Allocate::make(
        op->buffer_var, op->dtype,
        extents, condition, body,
        op->new_expr, op->free_function);
  }
  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_->dtype);
    Map<Var, Expr> values{{var_, idx}};
    stmt = Substitute(stmt, values);
    return For::make(idx, 0, var_lanes_, ForType::Serial, DeviceAPI::None, stmt);
  }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // ramp representing the var.
  Expr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  // The lets
  std::unordered_map<const Variable*, Expr> lets_;
  // mutate array, with given lane requirement
  // when finished, p_lane updates the lane requirement.
  Array<Expr> MutateArray(Array<Expr> arr, int* p_lanes) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<Expr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      Expr old_elem = arr[i];
      Expr new_elem = this->VisitExpr(old_elem);
      if (!new_elem.same_as(old_elem)) changed = true;
      new_arr[i] = new_elem;
      lanes = std::max(lanes, new_elem.dtype().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].dtype().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes);
        changed = true;
      }
    }
    if (!changed) return arr;
    return Array<Expr>(new_arr);
  }
  template<typename T>
  Expr BinaryVec(const T* op) {
    Expr a = this->VisitExpr(op->a);
    Expr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return GetRef<Expr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
  template<typename T>
  Expr AddSubVec(const T* op) {
    Expr a = this->VisitExpr(op->a);
    Expr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return GetRef<Expr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const Ramp* b_ramp = b.as<Ramp>();
        const Ramp* a_ramp = a.as<Ramp>();
        if (a.dtype().lanes() == 1 && b_ramp) {
          return Ramp::make(
              arith::Compute<T>(a, b_ramp->base),
              arith::Compute<T>(make_zero(b_ramp->stride.dtype()), b_ramp->stride),
              b_ramp->lanes);
        }
        if (b.dtype().lanes() == 1 && a_ramp) {
          return Ramp::make(
              arith::Compute<T>(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
};

class LoopVectorizer : public StmtMutator {
 public:
  Stmt VisitStmt_(const For* op) final {
    if (op->for_type == ForType::Vectorized) {
      CHECK(is_zero(op->min));
      int lanes = 0;
      bool succ = arith::GetConstInt(op->extent, &lanes);
      if (!succ || lanes < 1) {
        LOG(FATAL) << "Failed to vectorize loop with extent " << op->extent;
      }
      return Vectorizer(op->loop_var, lanes)(op->body);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

Stmt VectorizeLoop(Stmt stmt) {
  return LoopVectorizer()(std::move(stmt));
}

class VectorizeSkipper : public StmtMutator {
 public:
  Stmt VisitStmt_(const For* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<For>();
    if (op->for_type == ForType::Vectorized) {
      return For::make(op->loop_var, op->min, op->extent, ForType::Serial, op->device_api,
                       op->body);
    } else {
       return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) {
  return VectorizeSkipper()(std::move(stmt));
}

}  // namespace ir
}  // namespace tvm
