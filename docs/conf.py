#!/usr/bin/env python
#

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import verilog_ml_benchmark_generator

# -- General configuration ---------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx_click.ext']
source_suffix = '.rst'
master_doc = 'index'
project = 'Verilog ML Benchmark Generator'
copyright = "2020, Esther Roorda"
author = "Esther Roorda"
version = verilog_ml_benchmark_generator.__version__
release = verilog_ml_benchmark_generator.__version__
language = None
pygments_style = 'sphinx'
todo_include_todos = False
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------
html_theme = 'bizstyle'

# -- Options for HTMLHelp output ---------------------------------------
# Output file base name for HTML help builder.
htmlhelp_basename = 'verilog_ml_benchmark_generatordoc'

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'verilog_ml_benchmark_generator.tex',
     'Verilog ML Benchmark Generator Documentation',
     'Esther Roorda', 'manual'),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'verilog_ml_benchmark_generator',
     'Verilog ML Benchmark Generator Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'verilog_ml_benchmark_generator',
     'Verilog ML Benchmark Generator Documentation',
     author,
     'verilog_ml_benchmark_generator',
     'One line description of project.',
     'Miscellaneous'),
]



