/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FINN XSI++: C++ XSI Binding used by FINN.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

#include "xsi_finn.hpp"

#include <algorithm>
#include <iostream>


using namespace xsi;

//===========================================================================
// Local Helpers


void Kernel::hex_in_lower() {
    for (unsigned i = 2; i < 4; i++)
        XZ10[i] |= ' ';
    for (unsigned i = 10; i < 16; i++)
        HEX[i] |= ' ';
}
void Kernel::hex_in_upper() {
    for (unsigned i = 2; i < 4; i++)
        XZ10[i] &= ~' ';
    for (unsigned i = 10; i < 16; i++)
        HEX[i] &= ~' ';
}
