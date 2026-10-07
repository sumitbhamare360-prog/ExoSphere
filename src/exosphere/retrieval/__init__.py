"""ExoSphere retrieval package."""

from __future__ import annotations

from .detection import (
    MOLECULE_INDICES,
    compute_all_bayes_factors,
    compute_bayes_factor,
    compute_upper_limits,
    detection_sigma,
    detection_summary,
    update_result_with_detection,
)
from .likelihood import (
    chi2,
    log_likelihood,
    log_likelihood_with_inflation,
)
from .priors import (
    DEFAULT_PRIOR_CONFIG,
    N_PARAMS,
    PARAM_NAMES,
    PriorConfig,
    check_vmr_sum,
    log_prior,
    make_prior_config_from_config,
    physical_to_unit,
    unit_to_physical,
)
from .results import RetrievalResult
from .samplers import (
    DEFAULT_SAMPLER_CONFIG,
    SamplerConfig,
    run,
    run_dynesty,
    run_jaxns,
    run_retrieval,
)

__all__ = [
    # priors
    "PriorConfig",
    "DEFAULT_PRIOR_CONFIG",
    "unit_to_physical",
    "physical_to_unit",
    "log_prior",
    "check_vmr_sum",
    "PARAM_NAMES",
    "N_PARAMS",
    "make_prior_config_from_config",
    # likelihood
    "log_likelihood",
    "log_likelihood_with_inflation",
    "chi2",
    # samplers
    "SamplerConfig",
    "DEFAULT_SAMPLER_CONFIG",
    "run_dynesty",
    "run_jaxns",
    "run_retrieval",
    "run",
    # results
    "RetrievalResult",
    # detection
    "MOLECULE_INDICES",
    "compute_bayes_factor",
    "compute_all_bayes_factors",
    "compute_upper_limits",
    "detection_sigma",
    "detection_summary",
    "update_result_with_detection",
]
