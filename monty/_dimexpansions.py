# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:02:55 2021

@author: atteb
"""

def broadcast(parameter : "parameter"):
    return parameter.shape[-1] if parameter.broadcasted else ()

def value(larger_than):
    def value(parameter : "parameter"):
        val = parameter.value
        return val if val > larger_than else ()
    return value
