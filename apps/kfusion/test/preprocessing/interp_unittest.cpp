#include "math_utils.h"
#include "gtest/gtest.h"

template <typename T>
class InterpTest: public ::testing::Test {
  protected:
    virtual void SetUp() {
      imgsize = make_uint2(64, 64);
      img = (T *) calloc (sizeof(T) * imgsize.x * imgsize.y, 1);
      for(unsigned y = 0; y < imgsize.y - 1; y++) 
        for (unsigned x = 0; x < imgsize.x - 1; x++) {
          // Checkboard
          img[x + y*imgsize.x] = (T)((x & 1) ^ (y & 1)); 
        }
    }

    uint2 imgsize;
    T * img; 
};

typedef ::testing::Types<float, double> testtypes;
TYPED_TEST_CASE(InterpTest, testtypes);

TYPED_TEST(InterpTest, OnOddPoint) {
  const float2 pix = make_float2(31.f, 32.f);
  const TypeParam interpolated = bilinear_interp(this->img, this->imgsize, pix);
  ASSERT_FLOAT_EQ(1.f, interpolated);
}

TYPED_TEST(InterpTest, OnEvenPoint) {
  const float2 pix = make_float2(24.f, 52.f);
  const TypeParam interpolated = bilinear_interp(this->img, this->imgsize, pix);
  ASSERT_FLOAT_EQ(0.f, interpolated);
}

TYPED_TEST(InterpTest, OnCellCentre) {
  const float2 pix = make_float2(24.5f, 52.5f);
  const TypeParam interpolated = bilinear_interp(this->img, this->imgsize, pix);
  ASSERT_FLOAT_EQ(0.5f, interpolated);
}

TYPED_TEST(InterpTest, OnCellQuarter) {
  const float2 pix = make_float2(24.25f, 52.25f);
  const TypeParam interpolated = bilinear_interp(this->img, this->imgsize, pix);
  ASSERT_FLOAT_EQ(0.375f, interpolated);
}

TYPED_TEST(InterpTest, OnCellArbitrary) {
  const float2 pix = make_float2(0.82f, 0.74f);
  const TypeParam interpolated = bilinear_interp(this->img, this->imgsize, pix);
  ASSERT_FLOAT_EQ(0.3464f, interpolated);
}
