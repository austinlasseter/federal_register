# from dotce import logger


def basic_scores(yy, yn, ny, nn):
    """
Params, count for each:

a) True Positive   yy >>    # of correct predictions that an observation is POSITIVE
b) False Negative  yn >>    # of incorrect predictions that an observation is NEGATIVE
c) False Positive  ny >>    # of incorrect predictions that an observation is POSITIVE
d) True Negative   nn >>    # of correct predictions that an observation is NEGATIVE

    AC = (yy+nn)/(yy+yn+ny+nn)
    TPR = yy/(yy+yn)
    FPR = ny/(ny+nn)
    TNR = nn/(ny+nn)
    FNR = yn/(yy+yn)
    PR = yy/(yy+ny)
    F = 2 * (PR*TPR) / (PR+TPR) # F1 score

    """
    yy = float(yy)
    nn = float(nn)
    yn = float(yn)
    ny = float(ny)
    try:
        out = dict(yy=yy,
                   yn=yn,
                   ny=ny,
                   nn=nn,
                   AC=(yy + nn) / (yy+yn+ny+nn),
                   TPR=yy / (yy + yn), # aka recall
                   FPR=ny / (ny + nn),
                   TNR=nn / (ny + nn),
                   FNR=yn / (yy + yn),
                   PR=yy / (yy + ny)) # precision
    except Exception as e:
        out = dict(yy=yy, nn=nn, yn=yn, ny=ny, AC=0.0, TPR=0.0, FPR=0.0, TNR=0.0, FNR=0.0, PR=0.0)
        # logger.error(str(e))
        # logger.error(str(out))
    out['F'] = 0.0
    try:
        out['F'] = 2.0 * (out['PR'] * out['TPR']) / (out['PR'] + out['TPR'])
    except Exception as e:
        # logger.error(str(e))
        print(e)
        print(out)
    # logger.debug(str(out))
    return out
