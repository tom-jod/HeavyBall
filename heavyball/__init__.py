from .foreach_adamw import ForeachAdamW
from .foreach_adopt import ForeachADOPT
from .foreach_laprop import ForeachLaProp
from .foreach_sfadamw import ForeachSFAdamW
from .foreach_soap import ForeachSOAP
from .p_adam import ForeachPaLMPAdam
from .palm_foreach_sfadamw import PaLMForeachSFAdamW
from .palm_foreach_soap import PaLMForeachSOAP
from .precond_schedule_foreach_soap import PrecondScheduleForeachSOAP
from .precond_schedule_palm_foreach_soap import PrecondSchedulePaLMForeachSOAP
from .precond_schedule_sfpsoap import PrecondScheduleSFPaLMSOAP
from .psgd_kron import ForeachPSGDKron
from .pure_psgd import ForeachPurePSGD
from .schedule_free_palm_foreach_soap import SFPaLMForeachSOAP

PalmForEachSoap = PaLMForeachSOAP

__all__ = ['PalmForEachSoap', 'PaLMForeachSFAdamW', 'PaLMForeachSOAP', 'SFPaLMForeachSOAP', 'PrecondScheduleSFPaLMSOAP',
           'ForeachSOAP', 'ForeachSFAdamW', 'ForeachLaProp', 'ForeachADOPT', 'PrecondScheduleForeachSOAP',
           'PrecondSchedulePaLMForeachSOAP', 'ForeachPSGDKron', 'ForeachAdamW', 'ForeachPurePSGD',
           'ForeachPaLMPAdam']
