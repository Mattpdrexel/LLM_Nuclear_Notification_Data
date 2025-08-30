import inspect
import trl
from trl.trainer import sft_config
from trl import SFTConfig
print('trl_version =', trl.__version__)
print('SFTConfig.__init__ signature:')
print(inspect.signature(SFTConfig.__init__))
print('\nFields in dataclass:')
for f in getattr(SFTConfig, '__dataclass_fields__', {}).values():
    print('-', f.name, '=', f.default)
print('\nModule path:', sft_config.__file__)
