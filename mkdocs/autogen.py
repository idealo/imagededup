# Heavily borrowed from the Auto-Keras project:
# https://github.com/jhfjhfj1/autokeras/blob/master/mkdocs/autogen.py

import ast
import os
import re


def delete_space(parts, start, end):
    if start > end or end >= len(parts):
        return None
    count = 0
    while count < len(parts[start]):
        if parts[start][count] == ' ':
            count += 1
        else:
            break
    return '\n'.join(y for y in [x[count:] for x in parts[start: end + 1] if len(x) > count])


def change_args_to_dict(string):
    if string is None:
        return None
    ans = []
    strings = string.split('\n')
    ind = 1
    start = 0
    while ind <= len(strings):
        if ind < len(strings) and strings[ind].startswith(" "):
            ind += 1
        else:
            if start < ind:
                ans.append('\n'.join(strings[start:ind]))
            start = ind
            ind += 1
    d = {}
    for line in ans:
        if (":" in line) and (line.count(':') == 1) and (len(line) > 0):
            lines = line.split(":")
            d[lines[0]] = lines[1].strip()
        elif (":" in line) and (line.count(':') > 1) and (len(line) > 0):
            lines = line.split(":")
            keyname = lines[0]
            val = ':'.join(lines[1:])
            d[keyname] = val
    return d


def remove_next_line(comments):
    for x in comments:
        if comments[x] is not None and '\n' in comments[x]:
            comments[x] = ' '.join(comments[x].split('\n'))
    return comments


def skip_space_line(parts, ind):
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            ind += 1
        else:
            break
    return ind


# check if comment is None or len(comment) == 0 return {}
def parse_func_string(comment):
    if comment is None or len(comment) == 0:
        return {}
    comments = {}
    paras = ('Args', 'Attributes', 'Returns', 'Raises', 'Example')
    comment_parts = [
        'short_description',
        'long_description',
        'Args',
        'Attributes',
        'Returns',
        'Raises',
        'Example',
    ]
    for x in comment_parts:
        comments[x] = None

    parts_init = re.split(r'\n', comment)

    parts = []

    for i in range(len(parts_init)):
        if parts_init[i] == 'Example:':
            parts.append(parts_init[i])
            code_part = '<sep>'.join([i.strip() for i in parts_init[len(parts):]]).replace('```', '')
            parts.append(code_part)
            break
        else:
            parts.append(parts_init[i])

    ind = 1
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            break
        else:
            ind += 1

    comments['short_description'] = '\n'.join(
        ['\n'.join(re.split('\n\s+', x.strip())) for x in parts[0:ind]]
    ).strip(':\n\t ')
    ind = skip_space_line(parts, ind)

    start = ind
    while ind < len(parts):
        if parts[ind].strip().startswith(paras):
            break
        else:
            ind += 1
    long_description = '\n'.join(
        ['\n'.join(re.split('\n\s+', x.strip())) for x in parts[start:ind]]
    ).strip(':\n\t ')
    comments['long_description'] = long_description

    ind = skip_space_line(paras, ind)
    while ind < len(parts):
        if parts[ind].strip().startswith(paras):
            start = ind
            start_with = parts[ind].strip()
            ind += 1
            while ind < len(parts):
                if parts[ind].strip().startswith(paras):
                    break
                else:
                    ind += 1
            part = delete_space(parts, start + 1, ind - 1)
            if start_with.startswith(paras[0]):
                comments[paras[0]] = change_args_to_dict(part)
            elif start_with.startswith(paras[1]):
                comments[paras[1]] = change_args_to_dict(part)
            elif start_with.startswith(paras[2]):
                comments[paras[2]] = change_args_to_dict(part)
            elif start_with.startswith(paras[3]):
                comments[paras[3]] = part
            elif start_with.startswith(paras[4]):
                comments[paras[4]] = part
            ind = skip_space_line(parts, ind)
        else:
            ind += 1

    remove_next_line(comments)
    return comments


def md_parse_line_break(comment):
    comment = comment.replace('  ', '\n\n')
    return comment.replace(' - ', '\n\n- ')


def to_md(comment_dict):
    doc = ''
    if 'short_description' in comment_dict:
        doc += comment_dict['short_description']
        doc += '\n\n'

    if 'long_description' in comment_dict:
        doc += md_parse_line_break(comment_dict['long_description'])
        doc += '\n'

    if 'Args' in comment_dict and comment_dict['Args'] is not None:
        doc += '##### Args\n'
        for arg, des in comment_dict['Args'].items():
            doc += '* **' + arg + '**: ' + des + '\n\n'

    if 'Attributes' in comment_dict and comment_dict['Attributes'] is not None:
        doc += '##### Attributes\n'
        for arg, des in comment_dict['Attributes'].items():
            doc += '* **' + arg + '**: ' + des + '\n\n'

    if 'Returns' in comment_dict and comment_dict['Returns'] is not None:
        doc += '##### Returns\n'
        if isinstance(comment_dict['Returns'], str):
            doc += comment_dict['Returns']
            doc += '\n'
        else:
            for arg, des in comment_dict['Returns'].items():
                doc += '* **' + arg + '**: ' + des + '\n\n'

    if 'Example' in comment_dict and comment_dict['Example'] is not None:
        doc += '##### Example usage\n'
        doc += '```python\n'
        if isinstance(comment_dict['Example'], str):
            for i in comment_dict['Example'].split('<sep>'):
                doc = doc + i
                doc += '\n'
        doc += '```\n'
    return doc


def parse_func_args(function):
    args = [a.arg for a in function.args.args if a.arg != 'self']
    kwargs = []
    if function.args.kwarg:
        kwargs = ['**' + function.args.kwarg.arg]

    return '(' + ', '.join(args + kwargs) + ')'


def get_func_comments(function_definitions):
    doc = ''
    for f in function_definitions:
        temp_str = to_md(parse_func_string(ast.get_docstring(f)))
        doc += ''.join(
            [
                '### ',
                f.name.replace('_', '\\_'),
                '\n',
                '```python',
                '\n',
                'def ',
                f.name,
                parse_func_args(f),
                '\n',
                '```',
                '\n',
                temp_str,
                '\n',
            ]
        )

    return doc


def get_comments_str(file_name):
    with open(file_name) as fd:
        file_contents = fd.read()
    module = ast.parse(file_contents)

    function_definitions = [node for node in module.body if (isinstance(node, ast.FunctionDef)) and (node.name[0] != '_' or node.name[:2] == '__')]

    doc = get_func_comments(function_definitions)

    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for class_def in class_definitions:
        temp_str = to_md(parse_func_string(ast.get_docstring(class_def)))

        # excludes private methods (start with '_')
        method_definitions = [
            node
            for node in class_def.body
            if isinstance(node, ast.FunctionDef) and (node.name[0] != '_' or node.name[:2] == '__')
        ]

        temp_str += get_func_comments(method_definitions)
        doc += '## class ' + class_def.name + '\n' + temp_str
    return doc


def extract_comments(directory):
    import sys
    print('python version:', sys.version)
    for parent, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            if os.path.splitext(file_name)[1] == '.py' and file_name != '__init__.py':
                # with open
                doc = get_comments_str(os.path.join(parent, file_name))
                directory = os.path.join('docs', parent.replace('../imagededup/', ''))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                output_file = open(os.path.join(directory, file_name[:-3] + '.md'), 'w')
                output_file.write(doc)
                output_file.close()


extract_comments('../imagededup/')
