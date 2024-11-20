from .cached_psgd_kron import ForeachCachedPSGDKron
from .delayed_psgd import ForeachDelayedPSGD
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
from .cached_delayed_psgd_kron import ForeachCachedDelayedPSGDKron

PalmForEachSoap = PaLMForeachSOAP

PaLMSOAP = PaLMForeachSOAP
PaLMSFAdamW = PaLMForeachSFAdamW
PaLMSFSoap = SFPaLMForeachSOAP
PrecondScheduleSFPaLMSOAP = PrecondScheduleSFPaLMSOAP
SOAP = ForeachSOAP
SFAdamW = ForeachSFAdamW
LaProp = ForeachLaProp
ADOPT = ForeachADOPT
PrecondScheduleSOAP = PrecondScheduleForeachSOAP
PrecondSchedulePaLMSOAP = PrecondSchedulePaLMForeachSOAP
PSGDKron = ForeachPSGDKron
AdamW = ForeachAdamW
PurePSGD = ForeachPurePSGD
PaLMPAdam = ForeachPaLMPAdam
DelayedPSGD = ForeachDelayedPSGD
CachedPSGDKron = ForeachCachedPSGDKron
CachedDelayedPSGDKron = ForeachCachedDelayedPSGDKron

__all__ = ['PalmForEachSoap', 'PaLMForeachSFAdamW', 'PaLMForeachSOAP', 'SFPaLMForeachSOAP', 'PrecondScheduleSFPaLMSOAP',
           'ForeachSOAP', 'ForeachSFAdamW', 'ForeachLaProp', 'ForeachADOPT', 'PrecondScheduleForeachSOAP',
           'PrecondSchedulePaLMForeachSOAP', 'ForeachPSGDKron', 'ForeachAdamW', 'ForeachPurePSGD', 'ForeachPaLMPAdam',
           'ForeachDelayedPSGD', 'ForeachCachedPSGDKron', 'ForeachCachedDelayedPSGDKron',  #
           'PaLMSOAP', 'PaLMSFAdamW', 'PaLMSFSoap', 'PaLMSFAdamW', 'PrecondScheduleSFPaLMSOAP',
           'SOAP', 'SFAdamW', 'LaProp', 'ADOPT', 'PSGDKron', 'AdamW', 'PurePSGD', 'PaLMPAdam', 'DelayedPSGD',
           'CachedPSGDKron', 'CachedDelayedPSGDKron', 'PrecondScheduleSOAP', 'PrecondSchedulePaLMSOAP']
