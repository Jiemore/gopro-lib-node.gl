/*
 * Copyright 2017 GoPro Inc.
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

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "bstr.h"
#include "log.h"
#include "nodegl.h"
#include "nodes.h"
#include "program.h"
#include "spirv.h"

#define OFFSET(x) offsetof(struct program_priv, x)
static const struct node_param computeprogram_params[] = {
#ifdef VULKAN_BACKEND
    // FIXME: should be the same param type on all backends
    {"compute", PARAM_TYPE_DATA, OFFSET(comp_data),
                .desc=NGLI_DOCSTRING("compute SPIR-V shader")},
#else
    {"compute", PARAM_TYPE_STR, OFFSET(compute), .flags=PARAM_FLAG_CONSTRUCTOR,
                .desc=NGLI_DOCSTRING("compute shader")},
#endif
    {NULL}
};

static int computeprogram_init(struct ngl_node *node)
{
    struct ngl_ctx *ctx = node->ctx;
    struct program_priv *s = node->priv_data;

#ifdef VULKAN_BACKEND
    return ngli_program_init(&s->program, ctx, NULL, 0, NULL, 0, s->comp_data, s->comp_data_size);
#else
    struct glcontext *gl = ctx->glcontext;
    if (!(gl->features & NGLI_FEATURE_COMPUTE_SHADER_ALL)) {
        LOG(ERROR, "context does not support compute shaders");
        return -1;
    }

    return ngli_program_init(&s->program, ctx, NULL, NULL, s->compute);
#endif
}

static void computeprogram_uninit(struct ngl_node *node)
{
    struct program_priv *s = node->priv_data;
    ngli_program_reset(&s->program);
}

const struct node_class ngli_computeprogram_class = {
    .id        = NGL_NODE_COMPUTEPROGRAM,
    .name      = "ComputeProgram",
    .init      = computeprogram_init,
    .uninit    = computeprogram_uninit,
    .priv_size = sizeof(struct program_priv),
    .params    = computeprogram_params,
    .file      = __FILE__,
};
