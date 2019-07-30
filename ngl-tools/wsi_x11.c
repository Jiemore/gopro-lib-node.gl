/*
 * Copyright 2018 GoPro Inc.
 *
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

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#include <nodegl.h>

#include "wsi.h"

int wsi_set_ngl_config(struct ngl_config *config, GLFWwindow *window)
{
#if 0
    Display *x11_display = glfwGetX11Display();
    Window x11_window = glfwGetX11Window(window);
    config->display = (uintptr_t)x11_display;
    config->window  = x11_window;
#endif

    //config->display = (uintptr_t)glfwGetWaylandDisplay();
    config->window = (uintptr_t)glfwGetWaylandWindow(window);

    return 0;
}
