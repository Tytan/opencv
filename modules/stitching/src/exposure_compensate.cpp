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

#include "precomp.hpp"

namespace cv {
namespace detail {

Ptr<ExposureCompensator> ExposureCompensator::createDefault(int type)
{
    if (type == NO)
        return makePtr<NoExposureCompensator>();
    if (type == GAIN)
        return makePtr<GainCompensator>(GainCompensator::GAIN);
    if (type == GAIN_BLOCKS)
        return makePtr<BlocksGainCompensator>(GainCompensator::GAIN);
    if (type == CHANNELS)
        return makePtr<GainCompensator>(GainCompensator::CHANNELS);
    if (type == CHANNELS_BLOCKS)
        return makePtr<BlocksGainCompensator>(GainCompensator::CHANNELS);
    CV_Error(Error::StsBadArg, "unsupported exposure compensation method");
}

void GainCompensator::feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
                           InputArrayOfArrays masks)
{
    LOGLN("Exposure compensation...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    const int num_images = images_.size().area();
    std::vector<UMat> images(num_images);
    Mat accumulated_gains;

    for (int i = 0; i < num_images; ++i)
        images_.getUMat(i).copyTo(images[i]);

    for (int  n = 0; n < num_feed; ++n)
    {
        single_feed(corners, images, masks);

        if (n == 0)
            accumulated_gains = gains_.clone();
        else
            multiply(accumulated_gains, gains_, accumulated_gains);

        if ((n+1) < num_feed)
            for (int i = 0; i < num_images; ++i)
                apply(i, corners[i], images[i], masks.getUMat(i));
    }
    gains_ = accumulated_gains;

    LOGLN("Exposure compensation, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}

void GainCompensator::single_feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
          InputArrayOfArrays masks_)
{
    std::vector<Mat> images, masks;
    images_.getMatVector(images);
    masks_.getMatVector(masks);

    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());
    Mat_<int> N(num_images, num_images); N.setTo(0);
    Mat I;
    if (mode == GAIN)
        I.create(num_images, num_images, CV_32F);
    else if (mode == CHANNELS)
        I.create(num_images, num_images, CV_32FC3);
    I.setTo(0);

    //Rect dst_roi = resultRoi(corners, images);
    Mat subimg1, subimg2;
    Mat_<uchar> submask1, submask2, intersect;

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = i; j < num_images; ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi))
            {
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i]));
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j]));

                submask1 = masks[i](Rect(roi.tl() - corners[i], roi.br() - corners[i]));
                submask2 = masks[j](Rect(roi.tl() - corners[j], roi.br() - corners[j]));
                intersect = submask1 & submask2;

                N(i, j) = N(j, i) = std::max(1, countNonZero(intersect));

                Vec3d Isum1 = 0, Isum2 = 0;
                for (int y = 0; y < roi.height; ++y)
                {
                    const Vec<uchar, 3>* r1 = subimg1.ptr<Vec<uchar, 3> >(y);
                    const Vec<uchar, 3>* r2 = subimg2.ptr<Vec<uchar, 3> >(y);
                    for (int x = 0; x < roi.width; ++x)
                    {
                        if (intersect(y, x))
                        {
                            if (mode == GAIN)
                            {
                                Isum1[0] += norm(r1[x]);
                                Isum2[0] += norm(r2[x]);
                            }
                            else if (mode == CHANNELS)
                            {
                                Isum1 += r1[x];
                                Isum2 += r2[x];
                            }
                        }
                    }
                }
                if (mode == GAIN)
                {
                    I.at<float>(i, j) = static_cast<float>(Isum1[0] / N(i, j));
                    I.at<float>(j, i) = static_cast<float>(Isum2[0] / N(i, j));
                }
                else if (mode == CHANNELS)
                {
                    I.at<Vec3f>(i, j) = static_cast<Vec3f>(Isum1 / N(i, j));
                    I.at<Vec3f>(j, i) = static_cast<Vec3f>(Isum2 / N(i, j));
                }
            }
        }
    }

    double alpha = 0.01;
    double beta = 100;

    for (int c = 0; c < I.channels(); ++c)
    {
        Mat_<float> A(num_images, num_images); A.setTo(0);
        Mat_<float> b(num_images, 1); b.setTo(0);
        for (int i = 0; i < num_images; ++i)
        {
            for (int j = 0; j < num_images; ++j)
            {
                b(i, 0) += beta * N(i, j);
                A(i, i) += beta * N(i, j);
                if (j != i)
                {
                    if (mode == GAIN)
                    {
                        A(i, i) += 2 * alpha * I.at<float>(i, j) * I.at<float>(i, j) * N(i, j);
                        A(i, j) -= 2 * alpha * I.at<float>(i, j) * I.at<float>(j, i) * N(i, j);
                    }
                    else if (mode == CHANNELS)
                    {
                        A(i, i) += 2 * alpha * I.at<Vec3f>(i, j)[c] * I.at<Vec3f>(i, j)[c] * N(i, j);
                        A(i, j) -= 2 * alpha * I.at<Vec3f>(i, j)[c] * I.at<Vec3f>(j, i)[c] * N(i, j);
                    }
                }
            }
        }

        Mat_<float> gains;
        solve(A, b, gains);

        gains_.create(num_images, 1, I.type());
        for (int i = 0; i < num_images; ++i)
        {
            if (mode == GAIN)
            {
                gains_.at<float>(i, 0) = gains(i, 0);
            }
            else if (mode == CHANNELS)
            {
                gains_.at<Vec3f>(i, 0)[c] = gains(i, 0);
            }
        }
    }
}


void GainCompensator::apply(int index, Point /*corner*/, InputOutputArray image, InputArray /*mask*/)
{
    CV_INSTRUMENT_REGION();

    if (mode == GAIN)
    {
        multiply(image, gains_.at<float>(index, 0), image);
    }
    else
    {
        Vec3f vec_gains = gains_.at<Vec3f>(index, 0);
        Scalar gains(vec_gains[0], vec_gains[1], vec_gains[2]);
        multiply(image, gains, image);
    }
}


Mat GainCompensator::gains() const
{
    Mat gains;
    gains_.copyTo(gains);
    return gains;
}


void BlocksGainCompensator::feed(const std::vector<Point> &corners, InputArrayOfArrays images_,
                                 InputArrayOfArrays masks_)
{
    std::vector<UMat> images, masks;
    images_.getUMatVector(images);
    masks_.getUMatVector(masks);
    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());

    std::vector<Size> bl_per_imgs(num_images);
    std::vector<Point> block_corners;
    std::vector<UMat> block_images;
    std::vector<UMat> block_masks;

    // Construct blocks for gain compensator
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
                        (images[img_idx].rows + bl_height_ - 1) / bl_height_);
        int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
        int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
        bl_per_imgs[img_idx] = bl_per_img;
        for (int by = 0; by < bl_per_img.height; ++by)
        {
            for (int bx = 0; bx < bl_per_img.width; ++bx)
            {
                Point bl_tl(bx * bl_width, by * bl_height);
                Point bl_br(std::min(bl_tl.x + bl_width, images[img_idx].cols),
                            std::min(bl_tl.y + bl_height, images[img_idx].rows));

                block_corners.push_back(corners[img_idx] + bl_tl);
                block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
                block_masks.push_back(masks[img_idx](Rect(bl_tl, bl_br)));
            }
        }
    }

    GainCompensator compensator(mode, num_feed);
    compensator.feed(block_corners, block_images, block_masks);
    Mat gains = compensator.gains();
    gain_maps_.resize(num_images);

    Mat_<float> ker(1, 3);
    ker(0,0) = 0.25; ker(0,1) = 0.5; ker(0,2) = 0.25;

    int bl_idx = 0;
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img = bl_per_imgs[img_idx];
        gain_maps_[img_idx].create(bl_per_img, gains.type());

        {
            Mat gain_map = gain_maps_[img_idx].getMat(ACCESS_WRITE);
            for (int by = 0; by < bl_per_img.height; ++by)
            {
                for (int bx = 0; bx < bl_per_img.width; ++bx, ++bl_idx)
                {
                    if (mode == GAIN)
                    {
                        gain_map.at<float>(by, bx) = gains.at<float>(bl_idx, 0);
                    }
                    else if (mode == CHANNELS)
                    {
                        gain_map.at<Vec3f>(by, bx) = gains.at<Vec3f>(bl_idx, 0);
                    }
                }
            }
        }
        sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
        sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
    }
}


void BlocksGainCompensator::apply(int index, Point /*corner*/, InputOutputArray _image, InputArray /*mask*/)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(_image.type() == CV_8UC3);

    UMat u_gain_map;
    if (gain_maps_[index].size() == _image.size())
        u_gain_map = gain_maps_[index];
    else
        resize(gain_maps_[index], u_gain_map, _image.size(), 0, 0, INTER_LINEAR);

    if (u_gain_map.channels() != 3)
    {
        std::vector<UMat> gains_channels;
        gains_channels.push_back(u_gain_map);
        gains_channels.push_back(u_gain_map);
        gains_channels.push_back(u_gain_map);
        merge(gains_channels, u_gain_map);
    }
    multiply(_image, u_gain_map, _image, 1, _image.type());
}

} // namespace detail
} // namespace cv
