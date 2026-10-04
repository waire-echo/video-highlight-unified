"""Override spatial settings while protecting input/output contracts."""

STAGES = {'YOLO 主体定位':'yolo', 'Pose 人物构图':'pose', 'BoT-SORT 主体锁定':'tracking'}
PROTECTED = {'input', 'output', 'base_predictions', 'video_dir', 'hints_dir', 'model', 'tracker',
             'cache', 'report', 'zip_output', 'device', 'seed', 'batch_size'}


def apply_parameters(commands, parameters):
    if set(parameters) - set(STAGES.values()):
        raise ValueError('Unknown spatial stage')
    for stage, command in commands:
        for name, value in parameters.get(STAGES[stage], {}).items():
            option = '--' + name.replace('_', '-')
            if name in PROTECTED or option not in command:
                raise ValueError('Unsupported spatial parameter: ' + name)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError('Spatial option must be a number or choice string: ' + name)
            command[command.index(option) + 1] = str(value)
    return commands
