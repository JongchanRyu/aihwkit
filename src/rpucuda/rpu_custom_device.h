/**
 * (C) Copyright 2020, 2021 IBM. All Rights Reserved.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#pragma once

#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

namespace RPU {

template <typename T> class CustomRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    Custom,
    /*implements*/
    DeviceUpdateType::Custom,
    /*parameter def*/
    T custom_coeff_up_1_a = (T)1.0;
    T custom_coeff_up_1_b = (T)1.0;
    T custom_coeff_up_1_c = (T)1.0;
    T custom_coeff_up_2_a = (T)1.0;
    T custom_coeff_up_2_b = (T)1.0;
    T custom_coeff_up_2_c = (T)1.0;
    T custom_coeff_up_3_a = (T)1.0;
    T custom_coeff_up_3_b = (T)1.0;
    T custom_coeff_up_3_c = (T)1.0;
    T custom_coeff_up_4_a = (T)1.0;
    T custom_coeff_up_4_b = (T)1.0;
    T custom_coeff_up_4_c = (T)1.0;
    T custom_coeff_up_5_a = (T)1.0;
    T custom_coeff_up_5_b = (T)1.0;
    T custom_coeff_up_5_c = (T)1.0;
    T custom_coeff_up_6_a = (T)1.0;
    T custom_coeff_up_6_b = (T)1.0;
    T custom_coeff_up_6_c = (T)1.0;
    T custom_coeff_down_1_a = (T)1.0;
    T custom_coeff_down_1_b = (T)1.0;
    T custom_coeff_down_1_c = (T)1.0;
    T custom_coeff_down_2_a = (T)1.0;
    T custom_coeff_down_2_b = (T)1.0;
    T custom_coeff_down_2_c = (T)1.0;
    T custom_coeff_down_3_a = (T)1.0;
    T custom_coeff_down_3_b = (T)1.0;
    T custom_coeff_down_3_c = (T)1.0;
    T custom_coeff_down_4_a = (T)1.0;
    T custom_coeff_down_4_b = (T)1.0;
    T custom_coeff_down_4_c = (T)1.0;
    T custom_coeff_down_5_a = (T)1.0;
    T custom_coeff_down_5_b = (T)1.0;
    T custom_coeff_down_5_c = (T)1.0;
    T custom_coeff_down_6_a = (T)1.0;
    T custom_coeff_down_6_b = (T)1.0;
    T custom_coeff_down_6_c = (T)1.0;
    T custom_coeff_up_a_dtod = (T)0.05;
    T custom_coeff_up_b_dtod = (T)0.05;
    T custom_coeff_up_c_dtod = (T)0.05;
    T custom_coeff_down_a_dtod = (T)0.05;
    T custom_coeff_down_b_dtod = (T)0.05;
    T custom_coeff_down_c_dtod = (T)0.05;

    int custom_num_sectors = 6;
    // '''
    // T ls_decrease_up = (T)0.0;
    // T ls_decrease_down = (T)0.0;
    // T ls_decrease_up_dtod = (T)0.05;
    // T ls_decrease_down_dtod = (T)0.05;

    // bool ls_allow_increasing_slope = false;
    // bool ls_mean_bound_reference = true;
    // bool ls_mult_noise = true;
    // '''
    ,
    /*print body*/

    ss << "There is no further parameters for custom device. - giho"  << std::endl;
    ,
    /* calc weight granularity body */
    return this->dw_min;
    ,
    /*Add*/
    bool implementsWriteNoise() const override { return true; };);


template <typename T> class CustomRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      CustomRPUDevice,
      /* ctor*/
      int x_sz = this->x_size_;
      int d_sz = this->d_size_;

      w_coeff_up_a_ = Array_3D_Get<T>(d_sz, x_sz, num_sectors_);
      w_coeff_up_b_ = Array_3D_Get<T>(d_sz, x_sz, num_sectors_);
      w_coeff_up_c_ = Array_3D_Get<T>(d_sz, x_sz, num_sectors_);
      w_coeff_down_a = Array_3D_Get<T>(d_sz, x_sz, num_sectors_);
      w_coeff_down_b = Array_3D_Get<T>(d_sz, x_sz, num_sectors_);
      w_coeff_down_c = Array_3D_Get<T>(d_sz, x_sz, num_sectors_);

      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          for (int k = 0; k < num_sectors_; ++k){
            w_coeff_up_a_[i][j][k] = (T)0.0;
            w_coeff_up_b_[i][j][k] = (T)0.0;
            w_coeff_up_c_[i][j][k] = (T)0.0;
            w_coeff_down_a_[i][j][k] = (T)0.0;
            w_coeff_down_b_[i][j][k] = (T)0.0;
            w_coeff_down_c_[i][j][k] = (T)0.0;
          }
        }
      }
      // '''int x_sz = this->x_size_;
      // int d_sz = this->d_size_;

      // w_slope_down_ = Array_2D_Get<T>(d_sz, x_sz);
      // w_slope_up_ = Array_2D_Get<T>(d_sz, x_sz);

      // for (int j = 0; j < x_sz; ++j) {
      //   for (int i = 0; i < d_sz; ++i) {
      //     w_slope_up_[i][j] = (T)0.0;
      //     w_slope_down_[i][j] = (T)0.0;
      //   }
      // }'''
    ,
      /* dtor*/
      Array_3D_Free<T>(coeff_up_a_,d_sz);
      Array_3D_Free<T>(coeff_up_b_,d_sz);
      Array_3D_Free<T>(coeff_up_c_,d_sz);
      Array_3D_Free<T>(coeff_down_a_,d_sz);
      Array_3D_Free<T>(coeff_down_b_,d_sz);
      Array_3D_Free<T>(coeff_down_c_,d_sz);
      // Array_2D_Free<T>(w_slope_down_);
      // Array_2D_Free<T>(w_slope_up_);
      ,
      /* copy */
      num_sectors_ = other.num_sectors_;
      for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          for (int k = 0; k < other.num_sectors_; ++k){
            w_coeff_up_a_[i][j][k] = other.w_coeff_up_a_[i][j][k];
            w_coeff_up_b_[i][j][k] = other.w_coeff_up_b_[i][j][k];
            w_coeff_up_c_[i][j][k] = other.w_coeff_up_c_[i][j][k];
            w_coeff_down_a_[i][j][k] = other.w_coeff_down_a_[i][j][k];
            w_coeff_down_b_[i][j][k] = other.w_coeff_down_b_[i][j][k];
            w_coeff_down_c_[i][j][k] = other.w_coeff_down_c_[i][j][k];
          }
        }
      }
      // '''for (int j = 0; j < other.x_size_; ++j) {
      //   for (int i = 0; i < other.d_size_; ++i) {
      //     w_slope_down_[i][j] = other.w_slope_down_[i][j];
      //     w_slope_up_[i][j] = other.w_slope_up_[i][j];
      //   }
      // }'''
    ,
      /* move assignment */
      num_sectors_ = other.num_sectors_;
      w_coeff_up_a_ = other.w_coeff_up_a_;
      w_coeff_up_b_ = other.w_coeff_up_b_;
      w_coeff_up_c_ = other.w_coeff_up_c_;
      w_coeff_down_a_ = other.w_coeff_down_a_;
      w_coeff_down_b_ = other.w_coeff_down_b_;
      w_coeff_down_c_ = other.w_coeff_down_c_;
      
      other.w_coeff_up_a_ = nullptr;
      other.w_coeff_up_b_ = nullptr;
      other.w_coeff_up_c_ = nullptr;
      other.w_coeff_down_a_ = nullptr;
      other.w_coeff_down_b_ = nullptr;
      other.w_coeff_down_c_ = nullptr;

      // '''w_slope_down_ = other.w_slope_down_;
      // w_slope_up_ = other.w_slope_up_;

      // other.w_slope_down_ = nullptr;
      // other.w_slope_up_ = nullptr;'''
      ,
      /* swap*/
      swap(a.num_sectors_, b.num_sectors_);
      swap(a.w_coeff_up_a_, b.w_coeff_up_a_);
      swap(a.w_coeff_up_b_, b.w_coeff_up_b_);
      swap(a.w_coeff_up_c_, b.w_coeff_up_c_);
      swap(a.w_coeff_down_a_, b.w_coeff_down_a_);
      swap(a.w_coeff_down_b_, b.w_coeff_down_b_);
      swap(a.w_coeff_down_c_, b.w_coeff_down_c_);

      // '''swap(a.w_slope_up_, b.w_slope_up_);
      // swap(a.w_slope_down_, b.w_slope_down_);'''
      ,
      /* dp names*/
      names.push_back(std::string("coeff_up_a"));
      names.push_back(std::string("coeff_up_b"));
      names.push_back(std::string("coeff_up_c"));
      names.push_back(std::string("coeff_down_a"));
      names.push_back(std::string("coeff_down_b"));
      names.push_back(std::string("coeff_down_c"));
      // names.push_back(std::string("num_sector"));
      // '''names.push_back(std::string("slope_up"));
      // names.push_back(std::string("slope_down"));'''
      ,
      /* dp2vec body*/
      #TODO
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_ * this->num_sectors_;

      for (int i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = w_coeff_up_a_[0][0][i];
        data_ptrs[n_prev + 1][i] = w_coeff_up_b_[0][0][i];
        data_ptrs[n_prev + 2][i] = w_coeff_up_c_[0][0][i];
        data_ptrs[n_prev + 3][i] = w_coeff_down_a_[0][0][i];
        data_ptrs[n_prev + 4][i] = w_coeff_down_b_[0][0][i];
        data_ptrs[n_prev + 5][i] = w_coeff_down_c_[0][0][i];
      }
      // '''int n_prev = (int)names.size();
      // int size = this->x_size_ * this->d_size_;

      // for (int i = 0; i < size; ++i) {
      //   data_ptrs[n_prev][i] = w_slope_up_[0][i];
      //   data_ptrs[n_prev + 1][i] = w_slope_down_[0][i];
      // }'''
    ,
      /* vec2dp body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_ * this->num_sectors_;

      for (int i = 0; i < size; ++i) {
        w_coeff_up_a_[0][0][i] = data_ptrs[n_prev][i];
        w_coeff_up_b_[0][0][i] = data_ptrs[n_prev + 1][i];
        w_coeff_up_c_[0][0][i] = data_ptrs[n_prev + 2][i];
        w_coeff_down_a_[0][0][i] = data_ptrs[n_prev + 3][i];
        w_coeff_down_b_[0][0][i] = data_ptrs[n_prev + 4][i];
        w_coeff_down_c_[0][0][i] = data_ptrs[n_prev + 5][i];
      }
      // '''int n_prev = (int)names.size();
      // int size = this->x_size_ * this->d_size_;

      // for (int i = 0; i < size; ++i) {
      //   w_slope_up_[0][i] = data_ptrs[n_prev][i];
      //   w_slope_down_[0][i] = data_ptrs[n_prev + 1][i];
      // }'''
    ,
      /*invert copy DP */
      num_sectors_ = rpu->getNumSectors();
      T ***coeff_up_a_ = rpu->getCoeffUpA();
      T ***coeff_up_b_ = rpu->getCoeffUpB();
      T ***coeff_up_c_ = rpu->getCoeffUpC();
      T ***coeff_down_a_ = rpu->getCoeffDownA();
      T ***coeff_down_b_ = rpu->getCoeffDownB();
      T ***coeff_down_c_ = rpu->getCoeffDownC();
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          for (int k = 0; k < num_sectors_; ++k){
            w_coeff_up_a_[i][j][k] = -coeff_up_a_[i][j][k];
            w_coeff_up_b_[i][j][k] = -coeff_up_b_[i][j][k];
            w_coeff_up_c_[i][j][k] = -coeff_up_c_[i][j][k];
            w_coeff_down_a_[i][j][k] = -coeff_down_a_[i][j][k];
            w_coeff_down_b_[i][j][k] = -coeff_down_b_[i][j][k];
            w_coeff_down_c_[i][j][k] = -coeff_down_c_[i][j][k];
          }
        }
      }
      // '''T **slope_down = rpu->getSlopeDown();
      // T **slope_up = rpu->getSlopeUp();
      // for (int j = 0; j < this->x_size_; ++j) {
      //   for (int i = 0; i < this->d_size_; ++i) {
      //     w_slope_down_[i][j] = -slope_up[i][j];
      //     w_slope_up_[i][j] = -slope_down[i][j];
      //   }
      // }'''
  );

  void printDP(int x_count, int d_count) const override;

  inline T ***getCoeffUpA() const { return w_coeff_up_a_; };
  inline T ***getCoeffUpB() const { return w_coeff_up_b_; };
  inline T ***getCoeffUpC() const { return w_coeff_up_c_; };
  inline T ***getCoeffDownA() const { return w_coeff_down_a_; };
  inline T ***getCoeffDownB() const { return w_coeff_down_b_; };
  inline T ***getCoeffDownC() const { return w_coeff_down_c_; };
  inline T getNumSectors() const { return num_sectors_; };
//   inline T **getSlopeUp() const { return w_slope_up_; };
//   inline T **getSlopeDown() const { return w_slope_down_; };

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

private:
  T ***w_coeff_up_a_ = nullptr;
  T ***w_coeff_up_b_ = nullptr;
  T ***w_coeff_up_c_ = nullptr;
  T ***w_coeff_down_a_ = nullptr;
  T ***w_coeff_down_b_ = nullptr;
  T ***w_coeff_down_c_ = nullptr;
  int num_sectors_ = 6;
//   T **w_slope_up_ = nullptr;
//   T **w_slope_down_ = nullptr;
};
} // namespace RPU
