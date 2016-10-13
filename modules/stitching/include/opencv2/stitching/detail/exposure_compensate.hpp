/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP
#define OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP

#if defined(NO)
#  warning Detected Apple 'NO' macro definition, it can cause build conflicts. Please, include this header before any Apple headers.
#endif

#include "opencv2/core.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching_exposure
//! @{

/** @brief Base class for all exposure compensators.
 */
class CV_EXPORTS ExposureCompensator
{
public:
    virtual ~ExposureCompensator() {}

    enum { NO, GAIN, GAIN_BLOCKS, GAIN_COMBINED, CHANNELS, CHANNELS_BLOCKS, CHANNELS_COMBINED };
    static Ptr<ExposureCompensator> createDefault(int type);

    /**
    @param corners Source image top-left corners
    @param images Source images
    @param masks Image masks to update
     */
    virtual void feed(const std::vector<Point> &corners, InputArrayOfArrays images,
                      InputArrayOfArrays masks) = 0;
    /** @brief Compensate exposure in the specified image.

    @param index Image index
    @param corner Image top-left corner
    @param image Image to process
    @param mask Image mask
     */
    virtual void apply(int index, Point corner, InputOutputArray image, InputArray mask) = 0;
};

/** @brief Stub exposure compensator which does nothing.
 */
class CV_EXPORTS NoExposureCompensator : public ExposureCompensator
{
public:
    void feed(const std::vector<Point> &/*corners*/, InputArrayOfArrays /*images*/,
              InputArrayOfArrays /*masks*/) CV_OVERRIDE { }
    void apply(int /*index*/, Point /*corner*/, InputOutputArray /*image*/, InputArray /*mask*/) CV_OVERRIDE { }
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image
intensities, see @cite BL07 and @cite WJ10 for details.
 */
class CV_EXPORTS GainCompensator : public ExposureCompensator
{
public:
    enum Mode { GAIN, CHANNELS };
    GainCompensator(Mode mode_=GAIN, int nfeed=3) : mode(mode_), num_feed(nfeed) {}
    void feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
              InputArrayOfArrays masks) CV_OVERRIDE;
    void single_feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
              InputArrayOfArrays masks_);
    void apply(int index, Point corner, InputOutputArray image, InputArray mask) CV_OVERRIDE;
    Mat gains() const;

protected:
    Mode mode;
    int num_feed;
    Mat gains_;
};

/** @brief Exposure compensator which tries to remove exposure related artifacts by adjusting image block
intensities, see @cite UES01 for details.
 */
class CV_EXPORTS BlocksGainCompensator : public GainCompensator
{
public:
    BlocksGainCompensator(Mode mode_=GAIN, int nfeed=3, int bl_width = 32, int bl_height = 32)
            : GainCompensator(mode_, nfeed), bl_width_(bl_width), bl_height_(bl_height) {}
    void feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
              InputArrayOfArrays masks_) CV_OVERRIDE;
    void apply(int index, Point corner, InputOutputArray image, InputArray mask) CV_OVERRIDE;

protected:
    int bl_width_, bl_height_;
    std::vector<UMat> gain_maps_;
};

/** @brief Exposure compensator which first compensate overall exposure,
 *  then compensate local exposure using a block-based algorithm
 */
class CV_EXPORTS CombinedGainCompensator : public BlocksGainCompensator
{
public:
    CombinedGainCompensator(Mode mode_=GAIN, int nfeed=3, int bl_width = 32, int bl_height = 32) :
        BlocksGainCompensator(mode_, nfeed, bl_width, bl_height) {}
    void feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
              InputArrayOfArrays masks);
};

//! @}

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_EXPOSURE_COMPENSATE_HPP
