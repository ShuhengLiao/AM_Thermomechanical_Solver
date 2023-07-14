import gamma.simulator.preprocessor as pre
import os

# Input Filename
geom_name = "fin_cube"
abq_file_name = os.path.join("geometries-toolpaths", geom_name, (geom_name+".inp"))
output_file_name = os.path.join("geometries-toolpaths", geom_name, ())

# Substrate thickness
subs_thic = 15.85 # mm

pre.write_keywords(abq_file_name, )