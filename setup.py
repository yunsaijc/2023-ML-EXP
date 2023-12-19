import os

path = os.path.dirname(__file__)
print(path)
bashrc_path = os.path.expanduser('~/.bashrc')

# 读取 ~/.bashrc 文件的内容
with open(bashrc_path, 'r') as file:
    lines = file.readlines()

# 检查 PYTHONPATH 是否已经包含这个路径
pythonpath_line = next((line for line in lines if line.startswith('export PYTHONPATH')), None)
if pythonpath_line and path in pythonpath_line:
    # print('Path is already in PYTHONPATH.')
    pass
else:
    # 在 ~/.bashrc 文件的末尾添加新的 PYTHONPATH
    lines.append(f'\nexport PYTHONPATH=$PYTHONPATH:{path}\n')

    # 写回 ~/.bashrc 文件
    with open(bashrc_path, 'w') as file:
        file.writelines(lines)
