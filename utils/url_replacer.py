#!/usr/bin/env python
import re

def replace_url(text):
    return re.sub(r'https?://\S+','网站链接', text)