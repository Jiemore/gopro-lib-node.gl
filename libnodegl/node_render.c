/*
 * Copyright 2016 GoPro Inc.
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

#include "hmap.h"
#include "log.h"
#include "nodegl.h"
#include "nodes.h"
<<<<<<< HEAD
#include "pass.h"
||||||| parent of e1b7e071... WIP: vulkan
=======
#include "spirv.h"
>>>>>>> e1b7e071... WIP: vulkan
#include "topology.h"
#include "utils.h"

struct render_priv {
    struct ngl_node *geometry;
    struct ngl_node *program;
    struct hmap *textures;
    struct hmap *uniforms;
    struct hmap *blocks;
    struct hmap *attributes;
    struct hmap *instance_attributes;
    int nb_instances;

    struct pass pass;
};

#define TEXTURES_TYPES_LIST (const int[]){NGL_NODE_TEXTURE2D,       \
                                          NGL_NODE_TEXTURE3D,       \
                                          NGL_NODE_TEXTURECUBE,     \
                                          -1}

#define PROGRAMS_TYPES_LIST (const int[]){NGL_NODE_PROGRAM,         \
                                          -1}

#define UNIFORMS_TYPES_LIST (const int[]){NGL_NODE_BUFFERFLOAT,     \
                                          NGL_NODE_BUFFERVEC2,      \
                                          NGL_NODE_BUFFERVEC3,      \
                                          NGL_NODE_BUFFERVEC4,      \
                                          NGL_NODE_UNIFORMFLOAT,    \
                                          NGL_NODE_UNIFORMVEC2,     \
                                          NGL_NODE_UNIFORMVEC3,     \
                                          NGL_NODE_UNIFORMVEC4,     \
                                          NGL_NODE_UNIFORMQUAT,     \
                                          NGL_NODE_UNIFORMINT,      \
                                          NGL_NODE_UNIFORMMAT4,     \
                                          NGL_NODE_ANIMATEDFLOAT,   \
                                          NGL_NODE_ANIMATEDVEC2,    \
                                          NGL_NODE_ANIMATEDVEC3,    \
                                          NGL_NODE_ANIMATEDVEC4,    \
                                          NGL_NODE_ANIMATEDQUAT,    \
                                          NGL_NODE_STREAMEDINT,     \
                                          NGL_NODE_STREAMEDFLOAT,   \
                                          NGL_NODE_STREAMEDVEC2,    \
                                          NGL_NODE_STREAMEDVEC3,    \
                                          NGL_NODE_STREAMEDVEC4,    \
                                          NGL_NODE_STREAMEDMAT4,    \
                                          -1}

#define ATTRIBUTES_TYPES_LIST (const int[]){NGL_NODE_BUFFERFLOAT,   \
                                            NGL_NODE_BUFFERVEC2,    \
                                            NGL_NODE_BUFFERVEC3,    \
                                            NGL_NODE_BUFFERVEC4,    \
                                            NGL_NODE_BUFFERMAT4,    \
                                            -1}

#define GEOMETRY_TYPES_LIST (const int[]){NGL_NODE_CIRCLE,          \
                                          NGL_NODE_GEOMETRY,        \
                                          NGL_NODE_QUAD,            \
                                          NGL_NODE_TRIANGLE,        \
                                          -1}

#define OFFSET(x) offsetof(struct render_priv, x)
static const struct node_param render_params[] = {
    {"geometry", PARAM_TYPE_NODE, OFFSET(geometry), .flags=PARAM_FLAG_CONSTRUCTOR,
                 .node_types=GEOMETRY_TYPES_LIST,
                 .desc=NGLI_DOCSTRING("geometry to be rasterized")},
    {"program",  PARAM_TYPE_NODE, OFFSET(program),
                 .node_types=PROGRAMS_TYPES_LIST,
                 .desc=NGLI_DOCSTRING("program to be executed")},
    {"textures", PARAM_TYPE_NODEDICT, OFFSET(textures),
                 .node_types=TEXTURES_TYPES_LIST,
                 .desc=NGLI_DOCSTRING("textures made accessible to the `program`")},
    {"uniforms", PARAM_TYPE_NODEDICT, OFFSET(uniforms),
                 .node_types=UNIFORMS_TYPES_LIST,
                 .desc=NGLI_DOCSTRING("uniforms made accessible to the `program`")},
    {"blocks",  PARAM_TYPE_NODEDICT, OFFSET(blocks),
                 .node_types=(const int[]){NGL_NODE_BLOCK, -1},
                 .desc=NGLI_DOCSTRING("blocks made accessible to the `program`")},
    {"attributes", PARAM_TYPE_NODEDICT, OFFSET(attributes),
                 .node_types=ATTRIBUTES_TYPES_LIST,
                 .desc=NGLI_DOCSTRING("extra vertex attributes made accessible to the `program`")},
    {"instance_attributes", PARAM_TYPE_NODEDICT, OFFSET(instance_attributes),
                 .node_types=ATTRIBUTES_TYPES_LIST,
                 .desc=NGLI_DOCSTRING("per instance extra vertex attributes made accessible to the `program`")},
    {"nb_instances", PARAM_TYPE_INT, OFFSET(nb_instances),
                 .desc=NGLI_DOCSTRING("number of instances to draw")},
    {NULL}
};

<<<<<<< HEAD
||||||| parent of e1b7e071... WIP: vulkan
static void draw_elements(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *indices = geometry->indices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glBindBuffer(gl, GL_ELEMENT_ARRAY_BUFFER, indices->buffer.id);
    ngli_glDrawElements(gl, gl_topology, indices->count, render->indices_type, 0);
}

static void draw_elements_instanced(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *indices = geometry->indices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glBindBuffer(gl, GL_ELEMENT_ARRAY_BUFFER, indices->buffer.id);
    ngli_glDrawElementsInstanced(gl, gl_topology, indices->count, render->indices_type, 0, render->nb_instances);
}

static void draw_arrays(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glDrawArrays(gl, gl_topology, 0, vertices->count);
}

static void draw_arrays_instanced(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glDrawArraysInstanced(gl, gl_topology, 0, vertices->count, render->nb_instances);
}

static int check_attributes(struct render_priv *s, struct hmap *attributes, int per_instance)
{
    if (!attributes)
        return 0;

    const struct hmap_entry *entry = NULL;
    while ((entry = ngli_hmap_next(attributes, entry))) {
        struct ngl_node *anode = entry->data;
        struct buffer_priv *buffer = anode->priv_data;

        if (per_instance) {
            if (buffer->count != s->nb_instances) {
                LOG(ERROR,
                    "attribute buffer %s count (%d) does not match instance count (%d)",
                    entry->key,
                    buffer->count,
                    s->nb_instances);
                return -1;
            }
        } else {
            struct geometry_priv *geometry = s->geometry->priv_data;
            struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
            if (buffer->count != vertices->count) {
                LOG(ERROR,
                    "attribute buffer %s count (%d) does not match vertices count (%d)",
                    entry->key,
                    buffer->count,
                    vertices->count);
                return -1;
            }
        }
    }
    return 0;
}

#define GEOMETRY_OFFSET(x) offsetof(struct geometry_priv, x)
static const struct {
    const char *const_name;
    int offset;
} attrib_const_map[] = {
    {"ngl_position", GEOMETRY_OFFSET(vertices_buffer)},
    {"ngl_uvcoord",  GEOMETRY_OFFSET(uvcoords_buffer)},
    {"ngl_normal",   GEOMETRY_OFFSET(normals_buffer)},
};

static int init_builtin_attributes(struct render_priv *s)
{
    s->pipeline_attributes = ngli_hmap_create();
    if (!s->pipeline_attributes)
        return -1;

    if (s->attributes) {
        const struct hmap_entry *entry = NULL;
        while ((entry = ngli_hmap_next(s->attributes, entry))) {
            struct ngl_node *anode = entry->data;
            int ret = ngli_hmap_set(s->pipeline_attributes, entry->key, anode);
            if (ret < 0)
                return ret;
        }
    }

    struct geometry_priv *geometry = s->geometry->priv_data;
    for (int i = 0; i < NGLI_ARRAY_NB(attrib_const_map); i++) {
        const int offset = attrib_const_map[i].offset;
        const char *const_name = attrib_const_map[i].const_name;
        uint8_t *buffer_node_p = ((uint8_t *)geometry) + offset;
        struct ngl_node *anode = *(struct ngl_node **)buffer_node_p;
        if (!anode)
            continue;

        int ret = ngli_hmap_set(s->pipeline_attributes, const_name, anode);
        if (ret < 0)
            return ret;
    }
    return 0;
}

=======
#ifndef VULKAN_BACKEND
static void draw_elements(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *indices = geometry->indices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glBindBuffer(gl, GL_ELEMENT_ARRAY_BUFFER, indices->buffer.id);
    ngli_glDrawElements(gl, gl_topology, indices->count, render->indices_type, 0);
}

static void draw_elements_instanced(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *indices = geometry->indices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glBindBuffer(gl, GL_ELEMENT_ARRAY_BUFFER, indices->buffer.id);
    ngli_glDrawElementsInstanced(gl, gl_topology, indices->count, render->indices_type, 0, render->nb_instances);
}

static void draw_arrays(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glDrawArrays(gl, gl_topology, 0, vertices->count);
}

static void draw_arrays_instanced(struct glcontext *gl, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
    const GLenum gl_topology = ngli_topology_get_gl_topology(geometry->topology);
    ngli_glDrawArraysInstanced(gl, gl_topology, 0, vertices->count, render->nb_instances);
}
#else
static void draw_elements(VkCommandBuffer cmd_buf, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *indices = geometry->indices_buffer->priv_data;
    vkCmdBindIndexBuffer(cmd_buf, indices->buffer.vkbuf, 0, render->indices_type);
    vkCmdDrawIndexed(cmd_buf, indices->count, 1, 0, 0, 0);
}

static void draw_elements_instanced(VkCommandBuffer cmd_buf, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *indices = geometry->indices_buffer->priv_data;
    vkCmdBindIndexBuffer(cmd_buf, indices->buffer.vkbuf, 0, render->indices_type);
    vkCmdDrawIndexed(cmd_buf, indices->count, render->nb_instances, 0, 0, 0);
}

static void draw_arrays(VkCommandBuffer cmd_buf, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
    vkCmdDraw(cmd_buf, vertices->count, 1, 0, 0);
}

static void draw_arrays_instanced(VkCommandBuffer cmd_buf, const struct render_priv *render)
{
    const struct geometry_priv *geometry = render->geometry->priv_data;
    const struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
    vkCmdDraw(cmd_buf, vertices->count, render->nb_instances, 0, 0);
}
#endif

static int check_attributes(struct render_priv *s, struct hmap *attributes, int per_instance)
{
    if (!attributes)
        return 0;

    const struct hmap_entry *entry = NULL;
    while ((entry = ngli_hmap_next(attributes, entry))) {
        struct ngl_node *anode = entry->data;
        struct buffer_priv *buffer = anode->priv_data;

        if (per_instance) {
            if (buffer->count != s->nb_instances) {
                LOG(ERROR,
                    "attribute buffer %s count (%d) does not match instance count (%d)",
                    entry->key,
                    buffer->count,
                    s->nb_instances);
                return -1;
            }
        } else {
            struct geometry_priv *geometry = s->geometry->priv_data;
            struct buffer_priv *vertices = geometry->vertices_buffer->priv_data;
            if (buffer->count != vertices->count) {
                LOG(ERROR,
                    "attribute buffer %s count (%d) does not match vertices count (%d)",
                    entry->key,
                    buffer->count,
                    vertices->count);
                return -1;
            }
        }
    }
    return 0;
}

#define GEOMETRY_OFFSET(x) offsetof(struct geometry_priv, x)
static const struct {
    const char *const_name;
    int offset;
} attrib_const_map[] = {
    {"ngl_position", GEOMETRY_OFFSET(vertices_buffer)},
    {"ngl_uvcoord",  GEOMETRY_OFFSET(uvcoords_buffer)},
    {"ngl_normal",   GEOMETRY_OFFSET(normals_buffer)},
};

static int init_builtin_attributes(struct render_priv *s)
{
    s->pipeline_attributes = ngli_hmap_create();
    if (!s->pipeline_attributes)
        return -1;

    if (s->attributes) {
        const struct hmap_entry *entry = NULL;
        while ((entry = ngli_hmap_next(s->attributes, entry))) {
            struct ngl_node *anode = entry->data;
            int ret = ngli_hmap_set(s->pipeline_attributes, entry->key, anode);
            if (ret < 0)
                return ret;
        }
    }

    struct geometry_priv *geometry = s->geometry->priv_data;
    for (int i = 0; i < NGLI_ARRAY_NB(attrib_const_map); i++) {
        const int offset = attrib_const_map[i].offset;
        const char *const_name = attrib_const_map[i].const_name;
        uint8_t *buffer_node_p = ((uint8_t *)geometry) + offset;
        struct ngl_node *anode = *(struct ngl_node **)buffer_node_p;
        if (!anode)
            continue;

        int ret = ngli_hmap_set(s->pipeline_attributes, const_name, anode);
        if (ret < 0)
            return ret;
    }
    return 0;
}

>>>>>>> e1b7e071... WIP: vulkan
static int render_init(struct ngl_node *node)
{
    struct ngl_ctx *ctx = node->ctx;
    struct render_priv *s = node->priv_data;
<<<<<<< HEAD
    struct pass_params params = {
||||||| parent of e1b7e071... WIP: vulkan

    /* Instancing checks */
    if (s->nb_instances && !(gl->features & NGLI_FEATURE_DRAW_INSTANCED)) {
        LOG(ERROR, "context does not support instanced draws");
        return -1;
    }

    if (s->instance_attributes && !(gl->features & NGLI_FEATURE_INSTANCED_ARRAY)) {
        LOG(ERROR, "context does not support instanced arrays");
        return -1;
    }

    int ret = check_attributes(s, s->attributes, 0);
    if (ret < 0)
        return ret;

    ret = check_attributes(s, s->instance_attributes, 1);
    if (ret < 0)
        return ret;

    if (!s->program) {
        s->program = ngl_node_create(NGL_NODE_PROGRAM);
        if (!s->program)
            return -1;
        int ret = ngli_node_attach_ctx(s->program, ctx);
        if (ret < 0)
            return ret;
    }

    struct geometry_priv *geometry = s->geometry->priv_data;
    if (geometry->indices_buffer) {
        int ret = ngli_node_buffer_ref(geometry->indices_buffer);
        if (ret < 0)
            return ret;
        s->has_indices_buffer_ref = 1;

        struct buffer_priv *indices = geometry->indices_buffer->priv_data;
        if (indices->block) {
            LOG(ERROR, "geometry indices buffers referencing a block are not supported");
            return -1;
        }

        ngli_format_get_gl_texture_format(gl, indices->data_format, NULL, NULL, &s->indices_type);
    }

    ret = init_builtin_attributes(s);
    if (ret < 0)
        return ret;

    struct pipeline_params params = {
=======

    /* Instancing checks */
#ifdef VULKAN_BACKEND
    /* TODO */
#else
    struct glcontext *gl = ctx->glcontext;

    if (s->nb_instances && !(gl->features & NGLI_FEATURE_DRAW_INSTANCED)) {
        LOG(ERROR, "context does not support instanced draws");
        return -1;
    }

    if (s->instance_attributes && !(gl->features & NGLI_FEATURE_INSTANCED_ARRAY)) {
        LOG(ERROR, "context does not support instanced arrays");
        return -1;
    }
#endif

    int ret = check_attributes(s, s->attributes, 0);
    if (ret < 0)
        return ret;

    ret = check_attributes(s, s->instance_attributes, 1);
    if (ret < 0)
        return ret;

    if (!s->program) {
        s->program = ngl_node_create(NGL_NODE_PROGRAM);
        if (!s->program)
            return -1;
        int ret = ngli_node_attach_ctx(s->program, ctx);
        if (ret < 0)
            return ret;
    }

    struct geometry_priv *geometry = s->geometry->priv_data;
    if (geometry->indices_buffer) {
        int ret = ngli_node_buffer_ref(geometry->indices_buffer);
        if (ret < 0)
            return ret;
        s->has_indices_buffer_ref = 1;

        struct buffer_priv *indices = geometry->indices_buffer->priv_data;
        if (indices->block) {
            LOG(ERROR, "geometry indices buffers referencing a block are not supported");
            return -1;
        }

#ifdef VULKAN_BACKEND
        const int index_node_type = geometry->indices_buffer->class->id;
        ngli_assert(index_node_type == NGL_NODE_BUFFERUSHORT ||
                    index_node_type == NGL_NODE_BUFFERUINT);
        s->indices_type = index_node_type == NGL_NODE_BUFFERUINT ? VK_INDEX_TYPE_UINT32
                                                                 : VK_INDEX_TYPE_UINT16;
#else
        ngli_format_get_gl_texture_format(gl, indices->data_format, NULL, NULL, &s->indices_type);
#endif
    }

    ret = init_builtin_attributes(s);
    if (ret < 0)
        return ret;

    struct pipeline_params params = {
>>>>>>> e1b7e071... WIP: vulkan
        .label = node->label,
<<<<<<< HEAD
        .geometry = s->geometry,
||||||| parent of e1b7e071... WIP: vulkan
=======
        .topology = geometry->topology,
>>>>>>> e1b7e071... WIP: vulkan
        .program = s->program,
        .textures = s->textures,
        .uniforms = s->uniforms,
        .blocks = s->blocks,
        .attributes = s->attributes,
        .instance_attributes = s->instance_attributes,
        .nb_instances = s->nb_instances,
    };
    return ngli_pass_init(&s->pass, ctx, &params);
}

static void render_uninit(struct ngl_node *node)
{
    struct render_priv *s = node->priv_data;
    ngli_pass_uninit(&s->pass);
}

static int render_update(struct ngl_node *node, double t)
{
    struct render_priv *s = node->priv_data;
<<<<<<< HEAD
    return ngli_pass_update(&s->pass, t);
||||||| parent of e1b7e071... WIP: vulkan

    int ret = ngli_node_update(s->geometry, t);
    if (ret < 0)
        return ret;

    return ngli_pipeline_update(&s->pipeline, t);
=======

    int ret = ngli_node_update(s->geometry, t);
    if (ret < 0)
        return ret;

#ifdef VULKAN_BACKEND
    struct ngl_ctx *ctx = node->ctx;
    ngli_honor_pending_glstate(ctx);
#endif

    return ngli_pipeline_update(&s->pipeline, t);
>>>>>>> e1b7e071... WIP: vulkan
}

static void render_draw(struct ngl_node *node)
{
<<<<<<< HEAD
||||||| parent of e1b7e071... WIP: vulkan
    struct ngl_ctx *ctx = node->ctx;
    struct glcontext *gl = ctx->glcontext;
=======
    struct ngl_ctx *ctx = node->ctx;
>>>>>>> e1b7e071... WIP: vulkan
    struct render_priv *s = node->priv_data;
<<<<<<< HEAD
    ngli_pass_exec(&s->pass);
||||||| parent of e1b7e071... WIP: vulkan

    int ret = ngli_pipeline_bind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "pipeline upload data error");
    }

    s->draw(gl, s);

    ret = ngli_pipeline_unbind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "could not unbind pipeline");
    }
=======

#ifdef VULKAN_BACKEND
    struct glcontext *vk = ctx->glcontext;

    VkCommandBuffer cmd_buf = s->pipeline.command_buffers[vk->img_index];

    VkCommandBufferBeginInfo command_buffer_begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };

    VkResult ret = vkBeginCommandBuffer(cmd_buf, &command_buffer_begin_info);
    if (ret != VK_SUCCESS)
        return;

    const float *rgba = vk->config.clear_color;
    VkClearValue clear_color = {.color.float32={rgba[0], rgba[1], rgba[2], rgba[3]}};
    VkRenderPassBeginInfo render_pass_begin_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = vk->render_pass,
        .framebuffer = vk->framebuffers[vk->img_index],
        .renderArea = {
            .extent = vk->extent,
        },
        .clearValueCount = 1,
        .pClearValues = &clear_color,
    };

    vkCmdBeginRenderPass(cmd_buf, &render_pass_begin_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {
        .width = vk->config.width,
        .height = vk->config.height,
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    vkCmdSetViewport(cmd_buf, 0, 1, &viewport);

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = {vk->config.width, vk->config.height},
    };
    vkCmdSetScissor(cmd_buf, 0, 1, &scissor);

    ngli_pipeline_bind(&s->pipeline);

    s->draw(cmd_buf, s);

    ngli_pipeline_unbind(&s->pipeline);

    vkCmdEndRenderPass(cmd_buf);

    ret = vkEndCommandBuffer(cmd_buf);
    if (ret != VK_SUCCESS)
        return;

    if (!ngli_darray_push(&vk->command_buffers, &cmd_buf))
        return;
#else
    struct glcontext *gl = ctx->glcontext;

    int ret = ngli_pipeline_bind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "pipeline upload data error");
    }

    s->draw(gl, s);

    ret = ngli_pipeline_unbind(&s->pipeline);
    if (ret < 0) {
        LOG(ERROR, "could not unbind pipeline");
    }
#endif
>>>>>>> e1b7e071... WIP: vulkan
}

const struct node_class ngli_render_class = {
    .id        = NGL_NODE_RENDER,
    .name      = "Render",
    .init      = render_init,
    .uninit    = render_uninit,
    .update    = render_update,
    .draw      = render_draw,
    .priv_size = sizeof(struct render_priv),
    .params    = render_params,
    .file      = __FILE__,
};
