# -*- coding: utf-8 -*-
from lib.constant import Tokens, IDs

def fill_bos_eos_token(tokens):
    return [Tokens.BOS.value] + tokens + [Tokens.EOS.value]

def fill_bos_eos_id(ids):
    return [IDs.BOS.value] + ids + [IDs.EOS.value]