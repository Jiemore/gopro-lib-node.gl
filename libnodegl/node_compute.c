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
#include <limits.h>

#include "glincludes.h"
#include "hmap.h"
#include "log.h"
#include "math_utils.h"
#include "nodegl.h"
#include "nodes.h"
#include "utils.h"

#define TEXTURES_TYPES_LIST (const int[]){NGL_NODE_TEXTURE2D,       \
                                          -1}

#define PROGRAMS_TYPES_LIST (const int[]){NGL_NODE_COMPUTEPROGRAM,  \
                                          -1}

#define UNIFORMS_TYPES_LIST (const int[]){NGL_NODE_UNIFORMFLOAT,    \
                                          NGL_NODE_UNIFORMVEC2,     \
                                          NGL_NODE_UNIFORMVEC3,     \
                                          NGL_NODE_UNIFORMVEC4,     \
                                          NGL_NODE_UNIFORMQUAT,     \
                                          NGL_NODE_UNIFORMINT,      \
                                          NGL_NODE_UNIFORMMAT4,     \
                                          -1}

#define OFFSET(x) offsetof(struct compute_priv, x)
static const struct node_param compute_params[] = {
    {"nb_group_x", PARAM_TYPE_INT,      OFFSET(nb_group_x), .flags=PARAM_FLAG_CONSTRUCTOR,
                   .desc=NGLI_DOCSTRING("number of work groups to be executed in the x dimension")},
    {"nb_group_y", PARAM_TYPE_INT,      OFFSET(nb_group_y), .flags=PARAM_FLAG_CONSTRUCTOR,
                   .desc=NGLI_DOCSTRING("number of work groups to be executed in the y dimension")},
    {"nb_group_z", PARAM_TYPE_INT,      OFFSET(nb_group_z), .flags=PARAM_FLAG_CONSTRUCTOR,
                   .desc=NGLI_DOCSTRING("number of work groups to be executed in the z dimension")},
    {"program",    PARAM_TYPE_NODE,     OFFSET(program),    .flags=PARAM_FLAG_CONSTRUCTOR, .node_types=PROGRAMS_TYPES_LIST,
                   .desc=NGLI_DOCSTRING("compute program to be executed")},
    {"textures",   PARAM_TYPE_NODEDICT, OFFSET(textures),   .node_types=TEXTURES_TYPES_LIST,
                   .desc=NGLI_DOCSTRING("input and output textures made accessible to the compute `program`")},
    {"uniforms",   PARAM_TYPE_NODEDICT, OFFSET(uniforms),   .node_types=UNIFORMS_TYPES_LIST,
                   .desc=NGLI_DOCSTRING("uniforms made accessible to the compute `program`")},
    {"blocks",     PARAM_TYPE_NODEDICT, OFFSET(blocks),     .node_types=(const int[]){NGL_NODE_BLOCK, -1},
                   .desc=NGLI_DOCSTRING("input and output blocks made accessible to the compute `program`")},
    {NULL}
};

static int compute_init(struct ngl_node *node)
{
    struct ngl_ctx *ctx = node->ctx;
    struct compute_priv *s = node->priv_data;

#ifdef VULKAN_BACKEND
    /* TODO */
#else
    struct glcontext *gl = ctx->glcontext;

    if (!(gl->features & NGLI_FEATURE_COMPUTE_SHADER_ALL)) {
        LOG(ERROR, "context does not support compute shaders");
        return -1;
    }

    if (s->nb_group_x > gl->max_compute_work_group_counts[0] ||
        s->nb_group_y > gl->max_compute_work_group_counts[1] ||
        s->nb_group_z > gl->max_compute_work_group_counts[2]) {
        LOG(ERROR,
            "compute work group size (%d, %d, %d) exceeds driver limit (%d, %d, %d)",
            s->nb_group_x,
            s->nb_group_y,
            s->nb_group_z,
            gl->max_compute_work_group_counts[0],
            gl->max_compute_work_group_counts[1],
            gl->max_compute_work_group_counts[2]);
        return -1;
    }
#endif

    struct pipeline_params params = {
        .label = node->label,
        .program = s->program,
        .textures = s->textures,
        .uniforms = s->uniforms,
        .blocks = s->blocks,
    };
    return ngli_pipeline_init(&s->pipeline, ctx, &params);
}

static void compute_uninit(struct ngl_node *node)
{
    struct compute_priv *s = node->priv_data;
    ngli_pipeline_uninit(&s->pipeline);
}

static int compute_update(struct ngl_node *node, double t)
{
    struct compute_priv *s = node->priv_data;
    return ngli_pipeline_update(&s->pipeline, t);
}

static void compute_draw(struct ngl_node *node)
{
#ifdef VULKAN_BACKEND
    struct ngl_ctx *ctx = node->ctx;
    struct glcontext *vk = ctx->glcontext;
    struct compute_priv *s = node->priv_data;
    struct pipeline *pipeline = &s->pipeline;

    VkCommandBuffer cmd_buf = pipeline->command_buffers[vk->img_index];

    VkCommandBufferBeginInfo command_buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };

    VkResult vkret = vkBeginCommandBuffer(cmd_buf, &command_buffer_begin_info);
    if (vkret != VK_SUCCESS)
        return;

    int ret = ngli_pipeline_bind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "could not bind pipeline");
    }

    vkCmdDispatch(cmd_buf, s->nb_group_x, s->nb_group_y, s->nb_group_z);

    ret = ngli_pipeline_unbind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "could not unbind pipeline");
    }

    vkret = vkEndCommandBuffer(cmd_buf);
    if (vkret != VK_SUCCESS)
        return;

    vk->command_buffers[vk->nb_command_buffers++] = cmd_buf;
#else
    struct ngl_ctx *ctx = node->ctx;
    struct glcontext *gl = ctx->glcontext;
    struct compute_priv *s = node->priv_data;

    int ret = ngli_pipeline_bind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "pipeline upload data error");
    }

    ngli_glMemoryBarrier(gl, GL_ALL_BARRIER_BITS);
    ngli_glDispatchCompute(gl, s->nb_group_x, s->nb_group_y, s->nb_group_z);
    ngli_glMemoryBarrier(gl, GL_ALL_BARRIER_BITS);

    ret = ngli_pipeline_unbind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "could not unbind pipeline");
    }
#endif
}

const struct node_class ngli_compute_class = {
    .id        = NGL_NODE_COMPUTE,
    .name      = "Compute",
    .init      = compute_init,
    .uninit    = compute_uninit,
    .update    = compute_update,
    .draw      = compute_draw,
    .priv_size = sizeof(struct compute_priv),
    .params    = compute_params,
    .file      = __FILE__,
};
