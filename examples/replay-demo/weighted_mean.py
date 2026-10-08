# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def weighted_mean(values, weights):
    return sum(v * w for v, w in zip(values, weights)) / len(values)
