import numpy as np

def db_to_lin(x):
    return 10**(x/10)

def lin_to_db(x):
    return 10*np.log10(x)

"""
define global constants - these values are fixed and independent of wavelength for now
"""

lam_op = 1550 # operating wavelength centre [nm]
f_op = 299792458/(lam_op*1e-9) # operating frequency [Hz]
#f_op = 193.5e12
c_0 = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
rsym = 100 # symbol rate [GBaud]
h_p = 6.63*1e-34  # Planck's constant [Js]
alpha_db = 0.2 # loss [dB/km]
disp = 17 # fibre dispersion [ps/nm/km]
l_sp = 100 # span length [km]
nch = 100 # number of channels
gamma = 1.2 # nonlinearity coefficient [/W/km]
nf = 4.5 # EDFA noise figure [dB]
n_span = 1 # number of spans
#alpha_lin = np.log((10**(al/10)))/2
alpha_lin = np.log(db_to_lin(alpha_db))/2
beta2 = (disp*(lam_op**2))/(2*np.pi*c_0) # dispersion coefficient at given wavelength [ps^2/km]
l_eff = (1 - np.exp(-2*alpha_lin*l_sp))/(2*alpha_lin)  # effective length [km]
l_effa = 1/(2*alpha_lin)  # the asymptotic effective length [km]
pch_dbm=-1
pch_lin = 1e-3*db_to_lin(pch_dbm)  # ^ [W]
bw_tot = (nch*rsym)/1e3 # total BW of Nyquist signal [THz]
# g_wdm = (pch_lin*nch)/(bw_tot*1e12) # flat-top value of PSD of signal [W/Hz]
nf_lin = db_to_lin(nf)
gain = alpha_db*l_sp
gain_lin = db_to_lin(gain)
alpha_neper = alpha_db/4.343 # alpha [Neper/km]

def calculate_per_channel_nsr_for_link(link_length,number_of_active_channels):
    n_sp = int(link_length / l_sp)
    # NOTE: eta should not be calculated for anything other than the full modulated bandwidth!!!
    eta_unif = calculate_etaunif_nikita(n_sp, gamma, l_eff, beta2, rsym, alpha_neper, nch)
    sig_ase_sq = calculate_sig_ase_nikita(n_sp, gain_lin, nf_lin, f_op, rsym)
    #nsr = (eta_unif + sig_ase_sq)/pch_lin
    snr = calculate_max_snr(sig_ase_sq,eta_unif)  # max snr per channel
    nsr = 1/snr # minimum NSR per channel
    nsr = nsr*number_of_active_channels
    return nsr

def calculate_capacity(path_length):
    """ calculate total capacity across all channels of a path
    """
    n_sp = int(path_length/l_sp)
    eta_unif = calculate_etaunif_nikita(n_sp, gamma, l_eff, beta2, rsym, alpha_neper, nch)
    sig_ase_sq = calculate_sig_ase_nikita(n_sp, gain_lin, nf_lin, f_op, rsym)
    capacity = calculate_throughput(rsym, sig_ase_sq, eta_unif)
    return capacity

def calculate_lightpath_capacity(path_length, wavelength):
    """
    calculate capacity of a lightpath - currently wavelength argument does nothing - can be used to do wavelength-dependence
    """
    n_sp = int(path_length/l_sp)
    eta_unif = calculate_etaunif_nikita(n_sp, gamma, l_eff, beta2, rsym, alpha_neper, nch)
    sig_ase_sq = calculate_sig_ase_nikita(n_sp, gain_lin, nf_lin, f_op, rsym)
    capacity = calculate_throughput(rsym, sig_ase_sq, eta_unif)
    return capacity
############################## FUNCTIONS FOR CHECKING PHYSICAL LAYER IMPLEMENTATION ##############################

def calculate_etaunif_nikita(n_span, gamma, l_eff, beta2, r_sym, alpha_ne, n_ch):
    '''
    calculate eta unif as defined by Nikita
    '''
    return 1e6*(8/27)*( (n_span*(gamma**2)*l_eff)/(np.pi*beta2*(r_sym**2)) ) * np.arcsinh(1e-6*( ((np.pi**2)*beta2*(n_ch**2)*(r_sym**2))/(2*alpha_ne)))
    #return (8/27)*((n_span*(gamma**2)*l_eff)/(np.pi*(beta2)*(r_sym**2))) * np.arcsinh((((np.pi**2)*beta2*(n_ch**2)*(r_sym**2))/(2*alpha_lin)))
def calculate_sig_ase_nikita(n_span, gain_lin, nf_lin, f_centre, r_sym):
    h_p = 6.63*1e-34  # Planck's constant [Js]
    #return n_span*(np.exp(alpha_neper*l_span) - 1)*nf_lin*h_p*f_centre*r_sym*1e9
    return n_span*(gain_lin - 1)*nf_lin*h_p*f_centre*r_sym*1e9
def calculate_throughput(r_sym, sig_ase, eta_unif):
    '''
    calculate throughput in Tb/s
    '''
    return 1e-3 * 2 * r_sym * np.log2( 1 + (1/3) * ( (4 / (sig_ase**2 * eta_unif))**(1/3) ) )

def calculate_max_snr(sig_ase, eta_unif):
    '''
    calculate the maximum SNR
    '''
    return (1/3) * ( (4 / (sig_ase**2 * eta_unif))**(1/3) )


############################## OLD VERSION (based on Pogg) ##############################

# def calculate_gnli_ny(g_wdm, bw_tot, l_effa, l_eff, gamma, beta2, n_span):
#     '''
#     calculate the NLI noise power spectral density in W/Hz (assumes total BW in THz, beta2 in ps^2/km, gamma in /W/km, Gwdm in W/Hz)
#     '''
#     gnli_0_ny = 1e24*(8/27)*(gamma**2)*(g_wdm**3)*(l_eff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*l_effa*(bw_tot**2)  ) )/(np.pi*beta2*l_effa ))
#     return gnli_0_ny*n_span  # incoherent addition of NLI noise
#
# def calculate_ase_tot(gain_lin, nf_lin, f_centre, r_sym, n_span):
#     '''
#     calculate the ASE noise power in [W] (assumes symbol rate in GBd and f in Hz)
#     '''
#     h_p = 6.63*1e-34  # Planck's constant [Js]
#     return nf_lin*h_p*f_centre*(gain_lin - 1)*r_sym*1e9*n_span
#
# def calculate_snr(g_nli, ase_tot, pch_lin, b_meas):
#     '''
#     calculate linear SNR
#     '''
#     # b_meas = BW over which noise is measured in units of GHz, b_meas = r_sym for Nyquist channels
#     return pch_lin/(ase_tot + g_nli*b_meas*1e9)
