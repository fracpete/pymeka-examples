# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# build_incremental.py
# Copyright (C) 2024 Fracpete (fracpete at waikato dot ac dot nz)

import os
import sys
import meka.core.jvm as jvm
import mekaexamples.helper as helper
from weka.core.converters import load_any_file
from meka.core.mlutils import prepare_data
from meka.classifiers import MultiLabelClassifier


jvm.start()

# load data
if len(sys.argv) < 2:
    data_file = os.path.join(helper.get_data_dir(), "Music.arff")
else:
    data_file = sys.argv[1]
helper.print_info("Loading: %s" % data_file)
data = load_any_file(data_file)
prepare_data(data)

# build classifier
helper.print_info("Build BR classifier")
br = MultiLabelClassifier(classname="meka.classifiers.multilabel.incremental.BRUpdateable")
br.build_classifier(data.copy_structure())
for inst in data:
    br.update_classifier(inst)
print(br.model)

jvm.stop()
