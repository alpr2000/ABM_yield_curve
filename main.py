import numpy as np
import numpy.random as random

"""
Gilt market agent based model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This model simulates five agents:

1. Government: Inititates government bond auctions

2. Central Bank: Set interest rates, engages in OMO in the advanced model

3. "Pension Funds": Representing LDI, the buy in auctions and operate in the secondary market

4. "Hedge Funds": Representing leveraged investors, they only operate in the secondary market, profit motivated

5. "Noise Traders": Only operating in the secondary market, they trade randomly, representing retail investors and other non-professional market participants

Step 1: (Every Month 1-12)
    Central Bank sets interest rates based on a Taylor rule and a stochastic inflation rate based on a simple AR(1) process

Step 2: (Every Month 1)
    Government initiates bond auction, pension funds buy in based on their demand function

Step 3: (Every Month)
    Secondary market trading occurs, with pension funds, hedge funds, and noise traders all participating based on their respective strategies

Step 4: (Optional)
    The central bank may engage in open market operations (OMO) which will be executed as auctions in the secondary market, with pension funds and hedge funds participating based on their demand functions

"""


