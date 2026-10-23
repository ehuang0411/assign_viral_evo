import numpy as np

def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Jukes-Cantor corrected genetic distance using Hamming distance.
        For example 1.163.

    """
    if len(reference_sequence) != len(distant_sequence):
        raise ValueError("Sequences must be aligned and have equal length.")

    # Ignore positions with an insertion or deletion in either sequence.
    comparable_pairs = [
        (ref_base, dist_base)
        for ref_base, dist_base in zip(reference_sequence, distant_sequence)
        if ref_base != "-" and dist_base != "-"
    ]

    n_comparable = len(comparable_pairs)

    if n_comparable == 0:
        raise ValueError("No comparable non-indel positions.")

    n_differences = sum(
        ref_base != dist_base
        for ref_base, dist_base in comparable_pairs
    )

    # Proportion of observed differences, ignoring indels.
    p = n_differences / n_comparable

    if p == 0:
        return 0.0

    if p >= 3 / 4:
        raise ValueError(
            "Jukes-Cantor distance is undefined when p >= 0.75."
        )

    # Jukes-Cantor corrected distance.
    return -0.75 * np.log(1 - (4 / 3) * p)

def kimura_two_parameter(reference_sequence: str, distant_sequence: str) -> float:
    """The Kimura Two Parameter correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Kimura corrected genetic distance using Hamming distance.
        For example 1.196.

    """
    if len(reference_sequence) != len(distant_sequence):
        raise ValueError("Sequences must be aligned and have equal length.")

    purines = {"A", "G"}
    pyrimidines = {"C", "T"}

    transitions = 0
    transversions = 0
    n_comparable = 0

    for ref_base, dist_base in zip(reference_sequence, distant_sequence):
        # Ignore indels.
        if ref_base == "-" or dist_base == "-":
            continue

        n_comparable += 1

        if ref_base == dist_base:
            continue

        ref_base = ref_base.upper()
        dist_base = dist_base.upper()

        is_transition = (
            (ref_base in purines and dist_base in purines)
            or (ref_base in pyrimidines and dist_base in pyrimidines)
        )

        if is_transition:
            transitions += 1
        else:
            transversions += 1

    if n_comparable == 0:
        raise ValueError("No comparable non-indel positions.")

    P = transitions / n_comparable
    Q = transversions / n_comparable

    denominator = 1 - 2 * P - Q

    if denominator <= 0 or (1 - 2 * Q) <= 0:
        raise ValueError(
            "Kimura two-parameter distance is undefined for these sequences."
        )

    return -0.5 * np.log(denominator) - 0.25 * np.log(1 - 2 * Q)