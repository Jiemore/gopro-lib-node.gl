#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 GoPro Inc.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import os
import os.path as op
import sys
import array
import colorsys
import random
import pynodegl as ngl
from pynodegl_utils.misc import scene
from pynodegl_utils.tests.debug import get_debug_points
from pynodegl_utils.tests.cmp_cuepoints import test_cuepoints


_CIRCLE_RADIUS = 0.5

_DEBUG_POSITIONS = False


fields_vert = '''
in vec4 ngl_position;
in vec2 ngl_uvcoord;

uniform mat4 ngl_modelview_matrix;
uniform mat4 ngl_projection_matrix;
uniform mat3 ngl_normal_matrix;

out vec2 var_uvcoord;

void main()
{
    gl_Position = ngl_projection_matrix * ngl_modelview_matrix * ngl_position;
    var_uvcoord = ngl_uvcoord;
}
'''

# TODO
block_std140_tpl = '''
layout(%(layout)s) uniform fields_block {
%(block_definition)s
} fields;
'''

block_std430_tpl = '''
layout(%(layout)s, binding=1) buffer fields_block {
%(block_definition)s
} fields;
'''

fields_frag = '''
precision highp float;

in vec2 var_uvcoord;
out vec4 frag_color;

uniform int nb_fields;

layout(%(layout)s, binding=1) buffer fields_block {
%(block_definition)s
} fields;

layout(%(layout)s, binding=2) buffer colors_block {
%(color_definition)s
} colors;

float in_rect(vec4 rect, vec2 pos)
{
    return (1.0 - step(pos.x, rect.x)) * step(pos.x, rect.x + rect.z) *
           (1.0 - step(pos.y, rect.y)) * step(pos.y, rect.y + rect.w);
}

%(func_definitions)s

void main()
{
    float w = 1.0;
    float h = 1.0 / float(nb_fields);
    vec3 res = %(func_calls)s;
    frag_color = vec4(res, 1.0);
}
'''

array_tpl = '''
vec3 get_color_%(field_name)s(float w, float h, float x, float y)
{
    float amount = 0.0;
    int len = fields.%(field_name)s.length();
    for (int i = 0; i < len; i++) {
        %(amount_code)s
    }
    return colors.%(field_name)s * amount;
}
'''

single_tpl = '''
vec3 get_color_%(field_name)s(float w, float h, float x, float y)
{
    float amount = 0.0;
    %(amount_code)s
    return colors.%(field_name)s * amount;
}
'''

common_int_tpl = '''
        amount += float(fields.%(field_name)s%(vec_field)s) / 255. * in_rect(rect_%(comp_id)d, var_uvcoord);
'''

common_flt_tpl = '''
        amount += fields.%(field_name)s%(vec_field)s * in_rect(rect_%(comp_id)d, var_uvcoord);
'''

rect_array_tpl = '''
        vec4 rect_%(comp_id)d = vec4(x + %(row)f * w / (%(nb_rows)f * float(len)) + float(i) * w / float(len),
                                     y + %(col)f * h / %(nb_cols)f,
                                     w / %(nb_rows)f / float(len),
                                     h / %(nb_cols)f);
'''

rect_single_tpl = '''
        vec4 rect_%(comp_id)d = vec4(x + %(col)f * w / %(nb_cols)f,
                                     y + %(row)f * h / %(nb_rows)f,
                                     w / %(nb_cols)f,
                                     h / %(nb_rows)f);
'''


# row, col, is_int
_type_spec = dict(
    float=    (1, 1, False),
    vec2=     (1, 2, False),
    vec3=     (1, 3, False),
    vec4=     (1, 4, False),
    mat4=     (4, 4, False),
    int=      (1, 1, True),
    ivec2=    (1, 2, True),
    ivec3=    (1, 3, True),
    ivec4=    (1, 4, True),
    quat_mat4=(4, 4, False),
    quat_vec4=(1, 4, False),
)


def _get_display_glsl_func(field_name, field_type, is_array=False):
    rows, cols, is_int = _type_spec[field_type]
    nb_comp = rows * cols

    tpl = array_tpl if is_array else single_tpl
    rect_tpl = rect_array_tpl if is_array else rect_single_tpl
    common_tpl = common_int_tpl if is_int else common_flt_tpl

    tpl_data = dict(field_name=field_name, nb_comp=nb_comp)

    amount_code = ''
    for row in range(rows):
        for col in range(cols):
            comp_id = row * cols + col

            tpl_data['col'] = col
            tpl_data['row'] = row
            tpl_data['nb_cols'] = cols
            tpl_data['nb_rows'] = rows

            if nb_comp == 16:
                tpl_data['vec_field'] = '[%d][%d]' % (col, row)
            elif nb_comp >= 1 and nb_comp <= 4:
                tpl_data['vec_field'] = '.' + "xyzw"[comp_id] if nb_comp != 1 else ''
            else:
                assert False

            if is_array:
                tpl_data['vec_field'] = '[i]' + tpl_data['vec_field']

            tpl_data['comp_id'] = comp_id
            amount_code += rect_tpl % tpl_data
            amount_code += common_tpl % tpl_data

    tpl_data['amount_code'] = amount_code

    return tpl % tpl_data


def _get_debug_point(rect):
    xpos = rect[0] + rect[2]/2.
    ypos = rect[1] + rect[3]/2.
    xpos = xpos * 2. - 1
    ypos = ypos * 2. - 1
    return xpos, ypos


def _get_debug_positions_from_fields(fields):
    debug_points = {}
    for i, field in enumerate(fields):
        array_len = field.get('len')
        nb_rows, nb_cols, _ = _type_spec[field['type']]

        field_points = []
        name = field['name']
        comp_id = 0

        w = 1.0
        h = 1.0 / float(len(fields))
        x = 0.0
        y = float(i) * h

        if array_len is None:
            for row in range(nb_rows):
                for col in range(nb_cols):
                    rect = (x + col * w / float(nb_cols),
                            y + row * h / float(nb_rows),
                            w / float(nb_cols),
                            h / float(nb_rows))
                    debug_points['{}_{}'.format(name, comp_id)] = _get_debug_point(rect)
                    comp_id += 1

        else:
            for i in range(array_len):
                for row in range(nb_rows):
                    for col in range(nb_cols):
                        rect = (x + row * w / (nb_rows * array_len) + i * w / float(array_len),
                                y + col * h / float(nb_cols),
                                w / float(nb_rows) / float(array_len),
                                h / float(nb_cols))
                        debug_points['{}_{}'.format(name, comp_id)] = _get_debug_point(rect)
                        comp_id += 1

    return debug_points


def _get_render(cfg, quad, fields, block_definition, color_definition, fields_block, colors_block, layout, debug_positions=False):

    func_calls = []
    func_definitions = []
    for i, field in enumerate(fields):
        is_array = 'len' in field
        func_calls.append('get_color_%s(w, h, 0.0, %f * h)' % (field['name'], i))
        func_definitions.append(_get_display_glsl_func(field['name'], field['type'], is_array=is_array))

    frag_data = dict(
        block_definition=block_definition,
        color_definition=color_definition,
        layout=layout,
        func_definitions='\n'.join(func_definitions),
        func_calls=' + '.join(func_calls),
    )

    shader_version = '310 es' if cfg.backend == 'gles' else '430'
    header = '#version %s\n' % shader_version

    fragment = header + fields_frag % frag_data

    program = ngl.Program(vertex=header + fields_vert, fragment=fragment) #, label=title)
    render = ngl.Render(quad, program)
    render.update_blocks(fields_block=fields_block, colors_block=colors_block)
    render.update_uniforms(nb_fields=ngl.UniformInt(len(fields)))

    if debug_positions:
        debug_points = _get_debug_positions_from_fields(fields)
        dbg_circles = get_debug_points(cfg, debug_points, text_size=(.2, .1))
        g = ngl.Group(children=(render, dbg_circles))
        return g

    return render


def _get_visual(cfg, block_definition, color_definition, fields_block, colors_block, fields, layout, area, title):
    title_h = 1 / 10.

    ax, ay, aw, ah = area
    title_node = ngl.Text(
        title,
        box_corner=(ax, ay + ah - title_h, 0),
        box_width=(aw, 0, 0),
        box_height=(0, title_h, 0),
        fg_color=(0, 0, 0, 1),
        bg_color=(1, 1, 1, 1),
        aspect_ratio=cfg.aspect_ratio,
    )

    text_group = ngl.Group()
    nb_fields = len(fields)
    field_h = (ah - title_h) / float(nb_fields)
    for i, field in enumerate(fields):
        field_hpos = nb_fields - i - 1
        text_node = ngl.Text('#%02d %s' % (field['num'], field['id']),
                             box_corner=(ax, ay + field_hpos * field_h, 0),
                             box_width=(aw / 2., 0, 0),
                             box_height=(0, field_h, 0),
                             fg_color=list(field['color']) + [1],
                             halign='left',
                             aspect_ratio=cfg.aspect_ratio)
        text_group.add_children(text_node)

    quad = ngl.Quad((ax + aw / 2., ay, 0), (aw / 2., 0, 0), (0, ah - title_h, 0))
    render = _get_render(cfg, quad, fields, block_definition, color_definition, fields_block, colors_block, layout)

    return ngl.Group(children=(title_node, text_group, render))


def _get_field(fields, target_field_id):
    ret = []
    for field_info in fields:
        field_id = '{category}_{type}'.format(**field_info)
        if field_id == target_field_id:
            ret.append(field_info)
    return ret


def _block_scene(cfg, field_id, seed, layout, debug_positions):
    cfg.duration = _ANIM_DURATION
    cfg.aspect_ratio = (1, 1)
    all_fields, fields_block, colors_block, block_definition, color_definition = _get_block_spec(seed, layout)
    fields = _get_field(all_fields, field_id)
    quad = ngl.Quad((-1, -1, 0), (2, 0, 0), (0, 2, 0))
    render = _get_render(cfg, quad, fields, block_definition, color_definition, fields_block, colors_block, layout, debug_positions=debug_positions)
    return render


def _gen_floats(n):
    return [(i + .5) / float(n) for i in range(n)]


def _gen_ints(n):
    return [int(i * 256) for i in _gen_floats(n)]


def _get_spec(i_count=6, f_count=7, v2_count=5, v3_count=9, v4_count=2, mat_count=3):
    f_list     = _gen_floats(f_count)
    v2_list    = _gen_floats(v2_count * 2)
    v3_list    = _gen_floats(v3_count * 3)
    v4_list    = _gen_floats(v4_count * 4)
    i_list     = _gen_ints(i_count)
    iv2_list   = [int(x * 256) for x in v2_list]
    iv3_list   = [int(x * 256) for x in v3_list]
    iv4_list   = [int(x * 256) for x in v4_list]
    mat4_list  = _gen_floats(mat_count * 4 * 4)
    one_f      = _gen_floats(1)[0]
    one_v2     = _gen_floats(2)
    one_v3     = _gen_floats(3)
    one_v4     = _gen_floats(4)
    one_i      = _gen_ints(1)[0]
    one_mat4   = _gen_floats(4 * 4)
    one_quat   = one_v4

    f_array    = array.array('f', f_list)
    v2_array   = array.array('f', v2_list)
    v3_array   = array.array('f', v3_list)
    v4_array   = array.array('f', v4_list)
    i_array    = array.array('i', i_list)
    iv2_array  = array.array('i', iv2_list)
    iv3_array  = array.array('i', iv3_list)
    iv4_array  = array.array('i', iv4_list)
    mat4_array = array.array('f', mat4_list)

    spec = []
    spec += [dict(name='f_%d'  % i, type='float',     category='single', data=one_f)    for i in range(f_count)]
    spec += [dict(name='v2_%d' % i, type='vec2',      category='single', data=one_v2)   for i in range(v2_count)]
    spec += [dict(name='v3_%d' % i, type='vec3',      category='single', data=one_v3)   for i in range(v3_count)]
    spec += [dict(name='v4_%d' % i, type='vec4',      category='single', data=one_v4)   for i in range(v4_count)]
    spec += [dict(name='i_%d'  % i, type='int',       category='single', data=one_i)    for i in range(i_count)]
    spec += [dict(name='m4_%d' % i, type='mat4',      category='single', data=one_mat4) for i in range(mat_count)]
    spec += [dict(name='qm_%d' % i, type='quat_mat4', category='single', data=one_quat) for i in range(mat_count)]
    spec += [dict(name='qv_%d' % i, type='quat_vec4', category='single', data=one_quat) for i in range(v4_count)]
    spec += [
        dict(name='t_f',    type='float',      category='array',           data=f_array,    len=f_count),
        dict(name='t_v2',   type='vec2',       category='array',           data=v2_array,   len=v2_count),
        dict(name='t_v3',   type='vec3',       category='array',           data=v3_array,   len=v3_count),
        dict(name='t_v4',   type='vec4',       category='array',           data=v4_array,   len=v4_count),
        dict(name='t_i',    type='int',        category='array',           data=i_array,    len=i_count),
        dict(name='t_iv2',  type='ivec2',      category='array',           data=iv2_array,  len=v2_count),
        dict(name='t_iv3',  type='ivec3',      category='array',           data=iv3_array,  len=v3_count),
        dict(name='t_iv4',  type='ivec4',      category='array',           data=iv4_array,  len=v4_count),
        dict(name='t_mat4', type='mat4',       category='array',           data=mat4_array, len=mat_count),
        dict(name='ab_f',   type='float',      category='animated_buffer', data=f_array,    len=f_count),
        dict(name='ab_v2',  type='vec2',       category='animated_buffer', data=v2_array,   len=v2_count),
        dict(name='ab_v3',  type='vec3',       category='animated_buffer', data=v3_array,   len=v3_count),
        dict(name='ab_v4',  type='vec4',       category='animated_buffer', data=v4_array,   len=v4_count),
        dict(name='quat_m4', type='quat_mat4', category='animated',        data=one_quat),
        dict(name='quat_v4', type='quat_vec4', category='animated',        data=one_quat),
    ]

    return spec


_ANIM_DURATION = 5.0


def _get_anim_kf(key_cls, data):
    t0, t1, t2 = 0, _ANIM_DURATION / 2., _ANIM_DURATION
    return [
        key_cls(t0, data),
        key_cls(t1, data[::-1]),
        key_cls(t2, data),
    ]


_SPECS = dict(
    single_float=         lambda data: ngl.UniformFloat(data),
    single_vec2=          lambda data: ngl.UniformVec2(data),
    single_vec3=          lambda data: ngl.UniformVec3(data),
    single_vec4=          lambda data: ngl.UniformVec4(data),
    single_int=           lambda data: ngl.UniformInt(data),
    single_mat4=          lambda data: ngl.UniformMat4(data),
    single_quat_mat4=     lambda data: ngl.UniformQuat(data, as_mat4=True),
    single_quat_vec4=     lambda data: ngl.UniformQuat(data, as_mat4=False),
    array_float=          lambda data: ngl.BufferFloat(data=data),
    array_vec2=           lambda data: ngl.BufferVec2(data=data),
    array_vec3=           lambda data: ngl.BufferVec3(data=data),
    array_vec4=           lambda data: ngl.BufferVec4(data=data),
    array_int=            lambda data: ngl.BufferInt(data=data),
    array_ivec2=          lambda data: ngl.BufferIVec2(data=data),
    array_ivec3=          lambda data: ngl.BufferIVec3(data=data),
    array_ivec4=          lambda data: ngl.BufferIVec4(data=data),
    array_mat4=           lambda data: ngl.BufferMat4(data=data),
    #animated_float=       lambda data: ngl.AnimatedFloat(keyframes=_get_anim_kf(ngl.AnimKeyFrameFloat, data)),
    #animated_vec2=        lambda data: ngl.AnimatedVec2(keyframes=_get_anim_kf(ngl.AnimKeyFrameVec2, data)),
    #animated_vec3=        lambda data: ngl.AnimatedVec3(keyframes=_get_anim_kf(ngl.AnimKeyFrameVec3, data)),
    #animated_vec4=        lambda data: ngl.AnimatedVec4(keyframes=_get_anim_kf(ngl.AnimKeyFrameVec4, data)),
    animated_buffer_float=lambda data: ngl.AnimatedBufferFloat(keyframes=_get_anim_kf(ngl.AnimKeyFrameBuffer, data)),
    animated_buffer_vec2= lambda data: ngl.AnimatedBufferVec2(keyframes=_get_anim_kf(ngl.AnimKeyFrameBuffer, data)),
    animated_buffer_vec3= lambda data: ngl.AnimatedBufferVec3(keyframes=_get_anim_kf(ngl.AnimKeyFrameBuffer, data)),
    animated_buffer_vec4= lambda data: ngl.AnimatedBufferVec4(keyframes=_get_anim_kf(ngl.AnimKeyFrameBuffer, data)),
    animated_quat_mat4=   lambda data: ngl.AnimatedQuat(keyframes=_get_anim_kf(ngl.AnimKeyFrameQuat, data), as_mat4=True),
    animated_quat_vec4=   lambda data: ngl.AnimatedQuat(keyframes=_get_anim_kf(ngl.AnimKeyFrameQuat, data), as_mat4=False),
)


def _get_field_id(f):
    t = f['type']
    t = t.split('_')[1] if t.startswith('quat') else t
    return '%-5s %s%s' % (t, f['name'], '[%d]' % f['len'] if 'len' in f else '')


_LAYOUTS = ('std140', 'std430')


def _get_block_spec(seed=0, layout=_LAYOUTS[0]):
    spec = _get_spec()

    # Always the same colors whatever the user seed
    random.seed(0)

    all_fields = []
    max_id_len = 0
    for field_info in spec:
        node_func = _SPECS['{category}_{type}'.format(**field_info)]
        node = node_func(field_info['data'])
        node.set_label(field_info['name'])
        field_info['node'] = node
        field_info['id'] = _get_field_id(field_info)
        max_id_len = max(len(field_info['id']), max_id_len)
        hue = random.uniform(0, 1)
        field_info['color'] = colorsys.hls_to_rgb(hue, 0.6, 1.0)
        all_fields.append(field_info)

    # Seed only defines the random for the position of the fields
    random.seed(seed)
    fields_pos = range(len(all_fields))
    random.shuffle(fields_pos)

    for i, pos in enumerate(fields_pos):
        all_fields[pos]['num'] = i

    color_fields = [ngl.UniformVec3(f['color']) for f in all_fields]
    block_fields = [all_fields[pos]['node'] for pos in fields_pos]

    colors_block = ngl.Block(fields=color_fields, layout=layout, label='colors_block')
    fields_block = ngl.Block(fields=block_fields, layout=layout, label='fields_block')

    color_definition = '\n'.join('vec3 %s;' % f['name'] for f in all_fields)
    block_definition = '\n'.join('%s;%s// #%02d' % (all_fields[pos]['id'], (max_id_len - len(all_fields[pos]['id'])) * ' ', all_fields[pos]['num']) for i, pos in enumerate(fields_pos))

    return all_fields, fields_block, colors_block, block_definition, color_definition


@scene(seed=scene.Range(range=[0, 100]),
       layout=scene.List(choices=_LAYOUTS))
def debug_block(cfg, seed=0, layout=_LAYOUTS[0]):

    cfg.duration = _ANIM_DURATION

    all_fields, fields_block, colors_block, block_definition, color_definition = _get_block_spec(seed, layout)

    fields_single   = filter(lambda f: f['category'] == 'single', all_fields)
    fields_array    = filter(lambda f: f['category'] == 'array', all_fields)
    fields_animated = filter(lambda f: f['category'].startswith('animated'), all_fields)
    field_specs = (
        (fields_single,   (-1/3., -1, 2/3., 2.), 'Single fields'),
        (fields_array,    ( 1/3.,  0, 2/3., 1.), 'Arrays'),
        (fields_animated, ( 1/3., -1, 2/3., 1.), 'Animated'),
    )

    g = ngl.Group()
    block_def_text = ngl.Text(
        '{} block:\n\n{}'.format(layout, block_definition),
        valign='top',
        box_corner=(-1, -1, 0),
        box_width=(2/3., 0, 0),
        box_height=(0, 2, 0),
        aspect_ratio=cfg.aspect_ratio,
    )
    g.add_children(block_def_text)

    for cat_fields, area, title in field_specs:
        visual_fields = _get_visual(cfg, block_definition, color_definition, fields_block, colors_block, cat_fields, layout, area, title)
        g.add_children(visual_fields)
    return g


def _get_debug_positions(field_id):
    all_fields, fields_block, colors_block, block_definition, color_definition = _get_block_spec() #seed, layout)
    fields = _get_field(all_fields, field_id)
    return _get_debug_positions_from_fields(fields)


def _get_block_function(field_id):
    nb_keyframes = 5 if 'animated' in field_id else 1

    @test_cuepoints(points=_get_debug_positions(field_id), nb_keyframes=nb_keyframes, debug_positions=False)
    @scene(seed=scene.Range(range=[0, 100]),
           layout=scene.List(choices=_LAYOUTS),
           debug_positions=scene.Bool())
    def scene_func(cfg, seed=0, layout=_LAYOUTS[0], debug_positions=True):
        return _block_scene(cfg, field_id, seed, layout, debug_positions)
    return scene_func


# TODO: need tests to try std430 as well
for field_id in sorted(_SPECS.keys()):
    globals()['block_{}'.format(field_id)] = _get_block_function(field_id)
