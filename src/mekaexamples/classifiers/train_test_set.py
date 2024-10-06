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

# train_test_set.py
# Copyright (C) 2024 Fracpete (fracpete at waikato dot ac dot nz)

import os
import sys
import meka.core.jvm as jvm
import mekaexamples.helper as helper
from weka.core.converters import load_any_file
from meka.core.mlutils import prepare_data
from meka.classifiers import MultiLabelClassifier, Evaluation


jvm.start()

# load data
if len(sys.argv) < 3:
    train_file = os.path.join(helper.get_data_dir(), "Music_train.arff")
    test_file = os.path.join(helper.get_data_dir(), "Music_test.arff")
else:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
helper.print_info("Loading train: %s" % train_file)
train = load_any_file(train_file)
prepare_data(train)
helper.print_info("Loading test: %s" % test_file)
test = load_any_file(test_file)
prepare_data(test)

# compatible?
msg = train.equal_headers(test)
if msg is not None:
    raise Exception(msg)

# configure classifier
br = MultiLabelClassifier(classname="meka.classifiers.multilabel.BR")

# train and evaluate
helper.print_info("Build BR classifier on %s" % train_file)
helper.print_info("Evaluate BR classifier on %s" % test_file)
result = Evaluation.evaluate_model(br, train, test, top="PCut1", vop="3")
print(result)

jvm.stop()
