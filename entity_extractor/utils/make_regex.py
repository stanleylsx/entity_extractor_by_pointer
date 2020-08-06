import re

def make_regex(line):
    """
    文本变成正则字符串
    :param line:
    :return:
    """
    return re.sub(r'([()+*?\[\]])', r'\\\1', line)