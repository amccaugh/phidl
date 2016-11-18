from __future__ import unicode_literals, division, print_function, absolute_import

import io
import glob
import os
#import pypandoc
#try:
#    pypandoc.get_pandoc_version()
#except:
#    print('pandoc installation not found, attempting to download it')
#    pypandoc.download_pandoc()
#    pypandoc.get_pandoc_version()

    
    
class ProcessStep(object):
    
    def __init__(self, recipe, description, gds_layer):
        self.recipe = recipe
        self.description = description
        self.gds_layer = gds_layer
        
        
class Process(object):
    
    def __init__(self, recipes_dir, gds_files = None):
        self.steps = []
        self.gds_files = gds_files
        self.recipes_dir = recipes_dir
        self.recipes = self.load_recipes_directory(recipes_dir)

    def add_step(self, recipe, description, gds_layer):
        if recipe not in self.recipes:
            raise ValueError('No step named "%s" exists in the loaded recipes. Please check %s' % (recipe,self.recipes_dir))
        new_step = ProcessStep(recipe, description, gds_layer)
        self.steps.append(new_step)

        
    def check_gds_layers(self):
        layers_used = []
        for s in self.steps:
            l = s.gds_layer
            if l in layers_used:
                print('Warning: Layer %s is being used multiple times (in recipe %s)' % (l, s.recipe))
            layers_used.append(l)
            
            
    def write_pdf(self, filename = None):
        footer = ''
        md_output = ''
        for n, s in enumerate(self.steps):
            header = """\n\\newpage\n\n# Processing step %s (%s)\n\n-----\n\n""" % (n+1, s.recipe)
            instructions = self.recipes[s.recipe]
            md_output += header + instructions + footer
            
        pypandoc.convert_text(source = md_output, format = 'markdown', to = 'pdf', 
                              outputfile='hello214214.pdf', extra_args=['-s'])
        return md_output
        
    
#    def load_recipes_multiple_single_file(self, filename):
#        with open(filename, 'r') as text_file:
#            sections = text_file.read().split('[//]: # ')[1:]
#
#        for s in sections:
#            first_line_index = s.index('\n')
#            name = s[:first_line_index].strip(')( ')
#            instructions = s[first_line_index:].strip()
#            if name in self.recipes: raise ValueError('Duplicate recipe name found ()')
#            self.recipes[name] = instructions
            
    def load_recipes_directory(self, directory):
        recipes = {}
        recipe_files = glob.glob(directory + '*.md')
        for filepath in recipe_files:
            name = os.path.basename(filepath)
            name = os.path.splitext(name)[0]
            with open(filepath, 'r') as f:
                instructions = f.read()
            recipes[name] = instructions
        return recipes



gds_files = [
'SE0004.gds',
'SE0005.gds',
'SE0006.gds',
'SE0007.gds',
        ]
          
P = Process(recipes_dir = 'C:/Users/anm16/amcc/Code/device-layout/phidl/recipes/', gds_files = gds_files)

# Create patterned gold and dielectric layer
P.add_step(recipe = 'stepper_setup_01', description = 'Stepper setup', gds_layer = None)
P.add_step(recipe = 'gold_liftoff_01', description = 'Gold ground', gds_layer = 3)
P.check_gds_layers()
md = P.write_pdf(1)

#P.add_step(recipe = 'sio2_01', description = 'SiO2 dielectric', gds_layer = None)
#
## Do
#P.add_step(name = 'wsi_deposition_01', description = 'WSi deposition for SNSPDs', gds_layer = None)
#P.add_step(name = 'wsi_nanowires_01', description = 'WSi nanowire patterning', gds_layer = 0)
#P.add_step(name = 'wsi_etch_01', description = 'WSi SNSPD', gds_layer = 0)
#P.add_step(name = 'gold_liftoff_01', description = 'Gold contacts', gds_layer = 3, gds_inverted = True)

#%% 

import io
import pypandoc
try:
    pypandoc.get_pandoc_version()
except:
    print('pandoc installation not found, attempting to download it')
    pypandoc.download_pandoc()
    pypandoc.get_pandoc_version()


filename = 'C:\\Users\\anm16\\amcc\\Code\\device-layout\\phidl\\testprocess\\processes.md'
with io.open(filename, 'r') as text_file:
    md = text_file.read()
pypandoc.convert_text(source = md, format = 'markdown', to = 'pdf', 
                      outputfile='hello214214.pdf', extra_args=['-s'])

#%% Testing pdfkit

#import pdfkit
#pdfkit.from_string(html, 'out32153.pdf')


#%% Testing load functionality


filename = r'C:\Users\anm16\amcc\Code\device-layout\phidl\testprocess\processes.md'
with open(filename, 'r') as text_file:
    sections = text_file.read().split('[//]: # ')[1:]

names = []
texts = []
for s in sections:
    first_line_index = s.index('\n')
    name = s[:first_line_index].strip(')( ')
    text = s[first_line_index:].strip()
    names.append(name)
    texts.append(text)
    
#%% Trying YAML


import yaml

filename = 'C:\\Users\\anm16\\amcc\\Code\\device-layout\\phidl\\testprocess\\processes.yaml'
with open(filename) as f:
    y = yaml.load(f)

pypandoc.convert_text(source = y['wsi_deposition_01']['instructions'], format = 'markdown', to = 'pdf', 
                      outputfile='hello63414214.pdf', extra_args=['-s'])




#%% Trying creating a .lyp file