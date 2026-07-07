# -*- coding: utf-8 -*-
"""
Geometry helpers used by detector_instrument_function.py to check whether a
particle track hits the vessel wall, the collimator opening/detector plane,
or the plasma core.

hit_detector and through_the_core are vectorized: instead of stepping
through a track one timestep at a time in a Python loop (checking hit_vessel/
line-plane-intersection/etc. per step), they evaluate the check across every
timestep at once with numpy and pick out the first index where it's true.
This is a straight port of the original per-step math (same formulas, same
"first index wins" semantics) -- just batched, since the per-step Python
loop was the actual cost, not the arithmetic itself.
"""

import numpy as np


def hit_vessel(point):
    """
    This function tests if the particle has hit the vessel wall.

    Parameters
    ----------
    point : np.array
        Current position of the particle.
        Format- [x,y,z] in meters

    Returns
    -------
    bool
        Whether the particle has hit the detector.
        True- Particle has hit the vessel wall.
        False- Particle has not hit the vessel wall.
    """

    # Radial position
    radPos = np.sqrt(point[0]**2 + point[1]**2)

    # Radial boundary of the vessel
    if radPos >= 0.45:
        return True
    # Axial boundary of the vessel
    if abs(point[2]) >= 1.44:
        return True

    # Base case
    return False


def through_the_core(particleTrack, coreRad):
    """
    This function checks if the particle went through the core of the plasma as
    defined by coreRad.
    If the particle went through the core, it also returns the particle track
    upto that point.

    Parameters
    ----------
    particleTrack : np.array
        Single particle track data.
    coreRad : float
        Radius of the core [m].

    Returns
    -------
    throughTheCore : bool
        Whether the particle went through the core.
        True- The particle went through the core.
        False- The particle did not go through the core.
    coreTrack : np.array
        Particle track data upto the point of leaving the core.
    """

    # Matches the original loop range: j in [0, len(particleTrack[0])-1)
    numCheck = particleTrack.shape[1] - 1
    if numCheck <= 0:
        return False, []

    radPos = np.sqrt(particleTrack[0, :numCheck]**2 + particleTrack[1, :numCheck]**2)

    # First index the particle enters the core
    inCoreMask = radPos <= coreRad
    if not np.any(inCoreMask):
        return False, []
    j1 = np.argmax(inCoreMask)

    # First index at/after j1 where it's back at or beyond coreRad
    exitMask = radPos[j1:] >= coreRad
    if not np.any(exitMask):
        return False, []
    j2 = j1 + np.argmax(exitMask)

    coreTrack = particleTrack[:, :j2 + 1]

    return True, coreTrack


def hit_detector(particleTrack, detPos, detNorm, detSize, epsilonPlane=1e-6, epsilonLine=1e-4):
    """
    This function checks if the particle hits the detector.
    If the particle hits the detector, this function also returns the position
    and velocity at the time of the hit.

    Parameters
    ----------
    particleTrack : np.array
        Singple particle track data.
    detPos : np.array
        Position of the center of the detector.
    detNorm : np.array
        Normal vector of the detector.
    detSize : float
        Size of the detector
    epsilonPlane : float, optional
        Tolerance for how parallel the segment can be to the plane before
        it's treated as non-intersecting. Default 1e-6.
    epsilonLine : float, optional
        Tolerance for how far the plane-intersection point can be from the
        segment before it's rejected. Default 1e-4.

    Returns
    -------
    detectorHit : bool
        Whether the particle hit the detector.
        True- The particle hit the detector.
        False- The particle did not hit the detector.
    hitPos : np.array
        Position of the hit.
        Default is [0,0,0]
    hitVel : np.array
        Velocity of the hit.
        Default is [0,0,0]
    hitTrack : np.array
        Particle track upto the detector hit.
    """

    noHit = (False, np.array([0, 0, 0]), np.array([0, 0, 0]), [])

    # Matches the original loop range: j in [0, len(particleTrack[0])-1)
    numCheck = particleTrack.shape[1] - 1
    if numCheck <= 0:
        return noHit

    points = particleTrack[0:3, :].T  # (numTimesteps, 3)

    # Vessel-wall check on each candidate point1 (indices 0..numCheck-1).
    # The original loop breaks (stops considering any further j) the moment
    # hit_vessel(point1) is True, so only indices before the first vessel
    # hit are eligible.
    r = np.sqrt(points[:numCheck, 0]**2 + points[:numCheck, 1]**2)
    vesselHit = (r >= 0.45) | (np.abs(points[:numCheck, 2]) >= 1.44)

    limit = np.argmax(vesselHit) if np.any(vesselHit) else numCheck
    if limit <= 0:
        return noHit

    p0 = points[:limit]
    p1 = points[1:limit + 1]

    with np.errstate(divide='ignore', invalid='ignore'):

        # Line-plane intersection for every candidate segment at once
        u = p1 - p0
        dot = u @ detNorm
        validPlane = np.abs(dot) > epsilonPlane

        denom = np.where(validPlane, dot, 1.0)
        w = p0 - detPos
        fac = -(w @ detNorm) / denom
        intPoints = p0 + u * fac[:, None]

        # Distance from the intersection point to the segment (same formula
        # as the original close_to_line, batched over rows)
        lineVec = p0 - p1
        pntVec = p0 - intPoints
        lineLen = np.linalg.norm(lineVec, axis=1)
        lineUnit = lineVec / lineLen[:, None]
        pntVecScaled = pntVec / lineLen[:, None]
        t = np.clip(np.einsum('ij,ij->i', lineUnit, pntVecScaled), 0.0, 1.0)
        nearest = lineVec * t[:, None]
        onLine = np.linalg.norm(nearest - pntVec, axis=1) <= epsilonLine

    # Distance from the intersection point to the detector center
    distLim = np.sqrt(detSize / np.pi)
    onDetector = np.linalg.norm(intPoints - detPos, axis=1) <= distLim

    hitMask = validPlane & onLine & onDetector
    if not np.any(hitMask):
        return noHit

    j = np.argmax(hitMask)  # first (earliest in time) valid hit

    hitPos = intPoints[j]
    hitVel = particleTrack[3:, j]
    hitTrack = particleTrack[:, j + 1]

    return True, hitPos, hitVel, hitTrack
