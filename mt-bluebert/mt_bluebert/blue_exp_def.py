from typing import Dict, Set

import yaml

from mt_bluebert.data_utils.task_def import TaskType, DataFormat, EncoderModelType
from mt_bluebert.data_utils.vocab import Vocabulary
from mt_bluebert.blue_metrics import BlueMetric


class BlueTaskDefs:
    def __init__(self, task_def_path):
        with open(task_def_path) as fp:
            self.task_def_dic = yaml.load(fp, yaml.FullLoader)

        self.label_mapper_map = {}  # type: Dict[str, Vocabulary]
        self.n_class_map = {}  # type: Dict[str, int]
        self.data_format_map = {}
        self.task_type_map = {}
        self.metric_meta_map = {}
        self.enable_san_map = {}
        self.dropout_p_map = {}
        self.split_names_map = {}
        self.encoder_type = None
        for task, task_def in self.task_def_dic.items():
            assert "_" not in task, "task name should not contain '_', current task name: %s" % task
            self.n_class_map[task] = task_def["n_class"]
            self.data_format_map[task] = DataFormat[task_def["data_format"]]
            self.task_type_map[task] = TaskType[task_def["task_type"]]
            self.metric_meta_map[task] = tuple(BlueMetric[metric_name] for metric_name in task_def["metric_meta"])
            self.enable_san_map[task] = task_def["enable_san"]
            if self.encoder_type is None:
                self.encoder_type = EncoderModelType[task_def["encoder_type"]]
            else:
                if self.encoder_type != EncoderModelType[task_def["encoder_type"]]:
                    raise ValueError('The shared encoder has to be the same.')

            if "labels" in task_def:
                label_mapper = Vocabulary(True)
                for label in task_def["labels"]:
                    label_mapper.add(label)
                self.label_mapper_map[task] = label_mapper
            else:
                self.label_mapper_map[task] = None

            if "dropout_p" in task_def:
                self.dropout_p_map[task] = task_def["dropout_p"]

            if 'split_names' in task_def:
                self.split_names_map[task] = task_def['split_names']
            else:
                self.split_names_map[task] = ["train", "dev", "test"]

    @property
    def tasks(self) -> Set[str]:
        return self.task_def_dic.keys()
