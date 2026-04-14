import math

import numpy as np
from numba import njit

INV_SQ2PI = 0.3989422804014327

@njit(fastmath=True, cache=True, inline='always')
def log_L_1G_njit(x, y, inv_e_y, A1, V1, exp1, B1):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq = (x[i] - V1) * (x[i] - V1)
        r = (y[i] - B1 - A1 * math.exp(dx_sq * exp1)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc

@njit(fastmath=True, cache=True, inline='always')
def log_L_2G_V22x_njit(x, y, inv_e_y, A21, A22, V2, exp1, exp2, B2):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq = (x[i] - V2) * (x[i] - V2)
        r = (y[i] - B2
                   - A21 * math.exp(dx_sq * exp1)
                   - A22 * math.exp(dx_sq * exp2)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc

@njit(fastmath=True, cache=True, inline='always')
def log_L_3G_V32xV33x_njit(x, y, inv_e_y, A31, A32, A33, V3, exp1, exp2, exp3, B3):
    acc = 0.0
    for i in range(x.shape[0]):
        dx_sq = (x[i] - V3) * (x[i] - V3)
        r = (y[i] - B3
             - A31 * math.exp(dx_sq * exp1)
             - A32 * math.exp(dx_sq * exp2)
             - A33 * math.exp(dx_sq * exp3)) * inv_e_y[i]
        acc += r * r
    return -0.5 * acc

@njit(fastmath=True, cache=True, inline='always')
def log_prob_1G_njit_linear(xx,yy,inv_e_y,
                            uF1,uV1,uS1,uB1,
                            lowF1,spnF1,
                            lowV1,spnV1,
                            loglowS1,logspnS1,
                            lowB1,spnB1):
    
    if (uF1<0.0 or uF1>1.0 or
        uV1<0.0 or uV1>1.0 or
        uS1<0.0 or uS1>1.0 or
        uB1<0.0 or uB1>1.0): return -np.inf
    
    F1 = lowF1 + spnF1*uF1
    V1 = lowV1 + spnV1*uV1
    S1 = math.exp(loglowS1 + logspnS1*uS1)
    B1 = lowB1 + spnB1*uB1
    
    invS1 = 1.0/S1
    A1 = F1*invS1 * INV_SQ2PI
    
    exp1 = -0.5*invS1*invS1
    
    logl = log_L_1G_njit(xx,yy,inv_e_y,A1,V1,exp1,B1)
    return logl

@njit(fastmath=True, cache=True, inline='always')
def log_prob_1G_B1x_njit_linear(xx,yy,inv_e_y,
                                uF1,uV1,uS1,B1,
                                lowF1,spnF1,
                                lowV1,spnV1,
                                loglowS1,logspnS1):
    
    if (uF1<0.0 or uF1>1.0 or
        uV1<0.0 or uV1>1.0 or
        uS1<0.0 or uS1>1.0): return -np.inf
    
    F1 = lowF1 + spnF1 * uF1
    V1 = lowV1 + spnV1 * uV1
    S1 = math.exp(loglowS1 + logspnS1*uS1)
    
    invS1 = 1.0/S1
    A1 = F1*invS1 * INV_SQ2PI
    
    exp1 = -0.5*invS1*invS1
    
    logl = log_L_1G_njit(xx,yy,inv_e_y,A1,V1,exp1,B1)
    return logl


@njit(fastmath=True, cache=True, inline='always')
def log_prob_2G_V22xB2x_njit_linear(xx,yy,inv_e_y,
                                    uF21,uF22,uV2,uS21,uS22,B2,
                                    lowF21,spnF21,
                                    lowF22,spnF22,
                                    lowV,spnV,
                                    lowS21,spnS21,
                                    loghihS22,deltav):
    
    if (uF21<0.0 or uF21>1.0 or
        uF22<0.0 or uF22>1.0 or
        uV2 <0.0 or uV2 >1.0 or
        uS21<0.0 or uS21>1.0 or
        uS22<0.0 or uS22>1.0): return -np.inf
    
    F21 = lowF21 + spnF21 * uF21
    F22 = lowF22 + spnF22 * uF22
    V2  = lowV + spnV  * uV2
    
    S21 = lowS21 + spnS21 * uS21
    if S21+deltav<=0: return -math.inf
    loglowS22 = math.log(S21+deltav)
    S22 = math.exp(loglowS22 + (loghihS22-loglowS22)*uS22)
    logJ_S22 = math.log(S22) + (loghihS22 - loglowS22)
    
    invS21 = 1.0/S21
    invS22 = 1.0/S22
    A21 = F21*invS21 * INV_SQ2PI
    A22 = F22*invS22 * INV_SQ2PI
    
    exp1 = -0.5 * invS21 * invS21
    exp2 = -0.5 * invS22 * invS22
    
    logl = log_L_2G_V22x_njit(xx,yy,inv_e_y,A21,A22,V2,exp1,exp2,B2) + logJ_S22
    return logl

@njit(fastmath=True, cache=True, inline='always')
def log_prob_2G_V22x_njit_linear(xx,yy,inv_e_y,
                                    uF21,uF22,uV2,uS21,uS22,uB2,
                                    lowF21,spnF21,
                                    lowF22,spnF22,
                                    lowV,spnV,
                                    lowS21,spnS21,
                                    loghihS22,
                                    lowB,spnB,deltav):
    
    if (uF21<0.0 or uF21>1.0 or
        uF22<0.0 or uF22>1.0 or
        uV2 <0.0 or uV2 >1.0 or
        uS21<0.0 or uS21>1.0 or
        uS22<0.0 or uS22>1.0 or
        uB2 <0.0 or uB2 >1.0): return -np.inf
    
    F21 = lowF21 + spnF21 * uF21
    F22 = lowF22 + spnF22 * uF22
    V2  = lowV + spnV  * uV2
    B2  = lowB + spnB  * uB2
    
    S21 = lowS21 + spnS21 * uS21
    if S21+deltav<=0: return -math.inf
    loglowS22 = math.log(S21+deltav)
    S22 = math.exp(loglowS22 + (loghihS22-loglowS22)*uS22)
    logJ_S22 = math.log(S22) + (loghihS22 - loglowS22)
    
    invS21 = 1.0/S21
    invS22 = 1.0/S22
    A21 = F21*invS21 * INV_SQ2PI
    A22 = F22*invS22 * INV_SQ2PI
    
    exp1 = -0.5 * invS21 * invS21
    exp2 = -0.5 * invS22 * invS22
    
    logl = log_L_2G_V22x_njit(xx,yy,inv_e_y,A21,A22,V2,exp1,exp2,B2) + logJ_S22
    return logl


@njit(fastmath=True, cache=True, inline='always')
def log_prob_3G_V32xV33xB3x_njit_linear(xx,yy,inv_e_y,
                               uF31,uF32,uF33,uV3,uS31,uS32,uS33,B3,
                               lowF31,spnF31,lowF32,spnF32,lowF33,spnF33,
                               lowV,spnV,
                               lowS31,
                               lowS32,spnS32,
                               loghihS33,deltav):
    
    if (uF31<0.0 or uF31>1.0 or
        uF32<0.0 or uF32>1.0 or
        uF33<0.0 or uF33>1.0 or 
        uV3 <0.0 or uV3 >1.0 or
        uS31<0.0 or uS31>1.0 or
        uS32<0.0 or uS32>1.0 or
        uS33<0.0 or uS33>1.0): return -math.inf
    
    F31 = lowF31 + spnF31 * uF31
    F32 = lowF32 + spnF32 * uF32
    F33 = lowF33 + spnF33 * uF33
    
    V3  = lowV   + spnV  *uV3
    S32 = lowS32 + spnS32*uS32
    
    hihS31 = S32 - deltav
    # if S32+deltav<=0: return -math.inf
    loglowS33 = math.log(S32 + deltav)
    S31 = lowS31 + (hihS31-lowS31) * uS31
    S33 = math.exp(loglowS33 + (loghihS33-loglowS33)*uS33)
    
    logJ_S31 = math.log(S32-deltav-lowS31)
    logJ_S33 = math.log(S33) + (loghihS33-loglowS33)
    
    # S31 = lowS32 + spnS32*uS31
    # S33 = lowS32 + spnS32*uS33
    # if not (S31<S32<S33): return -math.inf    
    
    invS31 = 1.0/S31
    invS32 = 1.0/S32
    invS33 = 1.0/S33
    A31 = F31*invS31 * INV_SQ2PI
    A32 = F32*invS32 * INV_SQ2PI
    A33 = F33*invS33 * INV_SQ2PI
    
    exp1 = -0.5 * invS31 * invS31
    exp2 = -0.5 * invS32 * invS32
    exp3 = -0.5 * invS33 * invS33
    
    logl = log_L_3G_V32xV33x_njit(xx,yy,inv_e_y,
                                  A31,A32,A33,
                                  V3,
                                  exp1,exp2,exp3,
                                  B3) + logJ_S31 + logJ_S33
    return logl

@njit(fastmath=True, cache=True, inline='always')
def log_prob_3G_V32xV33x_njit_linear(xx,yy,inv_e_y,
                               uF31,uF32,uF33,uV3,uS31,uS32,uS33,uB3,
                               lowF31,spnF31,lowF32,spnF32,lowF33,spnF33,
                               lowV,spnV,
                               lowS31,
                               lowS32,spnS32,
                               loghihS33,
                               lowB,spnB,deltav):
    
    
    if (uF31<0.0 or uF31>1.0 or
        uF32<0.0 or uF32>1.0 or
        uF33<0.0 or uF33>1.0 or 
        uV3 <0.0 or uV3 >1.0 or
        uS31<0.0 or uS31>1.0 or
        uS32<0.0 or uS32>1.0 or
        uS33<0.0 or uS33>1.0 or
        uB3 <0.0 or uB3 >1.0): return -math.inf
    
    F31 = lowF31 + spnF31 * uF31
    F32 = lowF32 + spnF32 * uF32
    F33 = lowF33 + spnF33 * uF33
    
    V3  = lowV   + spnV*uV3
    B3  = lowB   + spnB*uB3
    S32 = lowS32 + spnS32 * uS32
    
    hihS31 = S32 - deltav
    # if S32+deltav<=0: return -math.inf
    loglowS33 = math.log(S32 + deltav)
    S31 = lowS31 + (hihS31-lowS31) * uS31
    S33 = math.exp(loglowS33 + (loghihS33-loglowS33)*uS33)
    
    logJ_S31 = math.log(S32-deltav-lowS31)
    logJ_S33 = math.log(S33) + (loghihS33-loglowS33)
    
    # S31 = lowS32 + spnS32*uS31
    # S33 = lowS32 + spnS32*uS33
    # if not (S31<S32<S33): return -math.inf    
    
    invS31 = 1.0/S31
    invS32 = 1.0/S32
    invS33 = 1.0/S33
    A31 = F31*invS31 * INV_SQ2PI
    A32 = F32*invS32 * INV_SQ2PI
    A33 = F33*invS33 * INV_SQ2PI

    exp1 = -0.5 * invS31 * invS31
    exp2 = -0.5 * invS32 * invS32
    exp3 = -0.5 * invS33 * invS33
    
    logl = log_L_3G_V32xV33x_njit(xx,yy,inv_e_y,
                                  A31,A32,A33,
                                  V3,
                                  exp1,exp2,exp3,
                                  B3) + logJ_S31 + logJ_S33
    return logl


# @njit(fastmath=True, cache=True, inline='always')
# def log_prob_3G_V32xV33xB3x_njit_sigsoft(xx,yy,inv_e_y,
#                                uA31,uA32,uA33,uV31,uS31,uS32,uS33,B3,
#                                low0,spn0,low3,spn3,low4,low5,spn5,deltav):
    
#     if abs(uA31)>8.0:  return -math.inf
#     if abs(uA32)>8.0:  return -math.inf
#     if abs(uA33)>8.0:  return -math.inf
#     if abs(uV31)>8.0:  return -math.inf
#     if abs(uS31)>8.0:  return -math.inf
#     if abs(uS32)>8.0:  return -math.inf
#     if abs(uS33)>20.0: return -math.inf
    
#     # if A33>A32: return -math.inf
#     A31,A32,A33,V31 = map_params_3G_A31A32A33V31(        uA31,uA32,uA33,uV31,low0,spn0,low3,spn3)
#     S31,S32,S33     = map_params_3G_S31S32S33_sigmoidsoftplus(uS31,uS32,uS33,low4,low5,spn5,deltav)
    
#     logl = log_L_3G_njit(xx,yy,inv_e_y,
#                          A31,A32,A33,
#                          V31,V31,V31,
#                          S31,S32,S33,
#                          B3)
    
#     return logl if math.isfinite(logl) else -math.inf

# @njit(fastmath=True, cache=True, inline='always')
# def log_prob_3G_V32xV33xB3x_njit_sigmoid(xx,yy,inv_e_y,
#                                uA31,uA32,uA33,uV31,uS31,uS32,uS33,B3,
#                                lowA,spnA,lowV,spnV,lowS31,lowS32,spnS32,hihS33,deltav):
    
#     # if uA33>uA32: return -math.inf
    
#     if abs(uA31)>8.0: return -math.inf
#     if abs(uA32)>8.0: return -math.inf
#     if abs(uA33)>8.0: return -math.inf
#     if abs(uV31)>8.0: return -math.inf
#     if abs(uS31)>8.0: return -math.inf
#     if abs(uS32)>8.0: return -math.inf
#     if abs(uS33)>8.0: return -math.inf
    
#     # if A33>A32: return -math.inf
#     A31,A32,A33,V31 = map_params_3G_A31A32A33V31(     uA31,uA32,uA33,uV31,lowA,spnA,lowV,spnV)
#     S31,S32,S33     = map_params_3G_S31S32S33_sigmoid(uS31,uS32,uS33,     lowS31,lowS32,spnS32,hihS33,deltav)
    
#     logl = log_L_3G_njit(xx,yy,inv_e_y,
#                          A31,A32,A33,
#                          V31,V31,V31,
#                          S31,S32,S33,
#                          B3)
    
#     return logl if math.isfinite(logl) else -math.inf

# @njit(fastmath=True, cache=True, inline='always')
# def log_prob_3G_V32xV33xB3x_njit_sigmoid2(xx,yy,inv_e_y,
#                                uA31,uA32,uA33,uV31,uS31,uS32,uS33,B3,
#                                lowA,spnA,
#                                lowV,spnV,
#                                lowS,spnS,lowS22,spnS22,
#                                deltav):
    
#     if abs(uA31)>8.0: return -math.inf
#     if abs(uA32)>8.0: return -math.inf
#     if abs(uA33)>8.0: return -math.inf
#     if abs(uV31)>8.0: return -math.inf
#     if abs(uS31)>8.0: return -math.inf
#     if abs(uS32)>8.0: return -math.inf
#     if abs(uS33)>8.0: return -math.inf
    
#     # if A33>A32: return -math.inf
#     A31,A32,A33,V31 = map_params_3G_A31A32A33V31(     uA31,uA32,uA33,uV31,lowA,spnA,lowV,spnV)
#     S31,S32,S33     = map_params_3G_S31S32S33_sigmoid2(uS31,uS32,uS33,lowS,spnS)
    
#     logl = log_L_3G_njit(xx,yy,inv_e_y,
#                          A31,A32,A33,
#                          V31,V31,V31,
#                          S31,S32,S33,
#                          B3)
    
#     return logl if math.isfinite(logl) else -math.inf


# @njit(fastmath=True, cache=True, inline='always')
# def log_prob_3G_V32xV33xB3x_njit_softplus(xx,yy,inv_e_y,
#                                uA31,uA32,uA33,uV31,uS31,uS32,uS33,B3,
#                                lowA,spnA,lowV,spnV,lowS31,deltav):
    
#     if abs(uA31)>8.0:  return -math.inf
#     if abs(uA32)>8.0:  return -math.inf
#     if abs(uA33)>8.0:  return -math.inf
#     if abs(uV31)>8.0:  return -math.inf
#     if abs(uS31)>20.0: return -math.inf
#     if abs(uS32)>20.0: return -math.inf
#     if abs(uS33)>20.0: return -math.inf
    
#     # if A33>A32: return -math.inf
#     A31,A32,A33,V31 = map_params_3G_A31A32A33V31( uA31,uA32,uA33,uV31,lowA,spnA,lowV,spnV)
#     S31,S32,S33     = map_params_3G_S31S32S33_softplus(uS31,uS32,uS33,lowS31,deltav)
    
#     logl = log_L_3G_njit(xx,yy,inv_e_y,
#                          A31,A32,A33,
#                          V31,V31,V31,
#                          S31,S32,S33,
#                          B3)
    
#     return logl if math.isfinite(logl) else -math.inf


# @njit(fastmath=True, cache=True, inline='always')
# def log_prob_3G_V32xV33xB3x_njit_tanh(xx,yy,inv_e_y,
#                                uA31,uA32,uA33,uV31,uS31,uS32,uS33,B3,
#                                cntrA,hspnA,
#                                cntrV,hspnV,
#                                lowS31,
#                                cntrS32,hspnS32,
#                                hihS33,deltav):
    
#     if uA33>uA32: return -math.inf
    
#     if (abs(uA31) > 5.0 or abs(uA32) > 5.0 or abs(uA33) > 5.0 or 
#         abs(uV31) > 5.0 or abs(uS31) > 5.0 or abs(uS32) > 5.0 or 
#         abs(uS33) > 5.0):
#         return -math.inf
    
#     tA31 = math.tanh(uA31)
#     tA32 = math.tanh(uA32)
#     tA33 = math.tanh(uA33)
#     tV31 = math.tanh(uV31)
#     tS32 = math.tanh(uS32)
#     tS31 = math.tanh(uS31)
#     tS33 = math.tanh(uS33)
    
#     A31 = cntrA + hspnA * tA31
#     A32 = cntrA + hspnA * tA32
#     A33 = cntrA + hspnA * tA33
#     V31 = cntrV + hspnV * tV31
#     S32 = cntrS32 + hspnS32 * tS32
    
#     half_diff_S31 = (S32 - deltav - lowS31) * 0.5
#     S31 = lowS31 + half_diff_S31 * (1.0 + tS31)
    
#     lowS33 = S32 + deltav
#     half_diff_S33 = (hihS33 - lowS33) * 0.5
#     S33 = lowS33 + half_diff_S33 * (1.0 + tS33)

#     logl = log_L_3G_njit(xx,yy,inv_e_y,
#                          A31,A32,A33,
#                          V31,V31,V31,
#                          S31,S32,S33,
#                          B3)
    
#     return logl if math.isfinite(logl) else -math.inf