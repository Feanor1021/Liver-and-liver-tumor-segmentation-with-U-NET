# Liver and Liver Tumor Segmentation with U-NET

This project aims to perform liver and liver tumor segmentation using the U-Net architecture. The U-Net is a convolutional neural network (CNN) architecture designed for semantic segmentation tasks, which involve classifying each pixel in an image into different classes or regions of interest.

## Table of Contents

- [Introduction](#introduction)
- [U-Net Architecture](#u-net-architecture)
- [Semantic Segmentation](#semantic-segmentation)

## Introduction

Liver and liver tumor segmentation is a crucial task in medical image analysis that can aid in diagnosing and treating liver diseases. In this project, we utilize the U-Net architecture for accurate and precise segmentation. The U-Net architecture has been widely used and proven effective for various medical image segmentation tasks.

## U-Net Architecture

The U-Net architecture is named after its U-shaped network design. It consists of an encoder path and a decoder path. The encoder path captures the context and high-level features from the input image through successive down-sampling layers. The decoder path uses up-sampling layers to recover spatial information and generate the segmentation mask.

The skip connections between the encoder and decoder paths help preserve fine-grained details and enable the network to learn both global and local information. This makes U-Net well-suited for tasks that require precise segmentation, such as liver and tumor segmentation.

## Semantic Segmentation

Semantic segmentation is a computer vision task that involves assigning a class label to each pixel in an image. It goes beyond simple object detection and aims to classify and delineate different regions or objects of interest within an image. In the context of this project, semantic segmentation is used to identify and segment the liver and liver tumor regions in medical CT scans.

