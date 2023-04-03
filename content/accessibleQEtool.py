#!/usr/bin/env python
# coding: utf-8

from math import pi
from IPython.display import SVG, display
import ipywidgets as ipw

class ClickToggleButtons(ipw.ToggleButtons):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_handlers = ipw.CallbackDispatcher()
        self.on_msg(self._handle_button_msg)
        pass

    def on_click(self, callback, remove=False):
        """Register a callback to execute when the button is clicked.

        The callback will be called with one argument, the clicked button
        widget instance.

        Parameters
        ----------
        remove: bool (optional)
            Set to true to remove the callback from the list of callbacks.
        """
        self._click_handlers.register_callback(callback, remove=remove)

    def _handle_button_msg(self, _, content, buffers):
        """Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg.
        """
        if content.get('event', '') == 'click':
            self._click_handlers(self)


def twelve_colors():
    # Copied from palettable.colorbrewer.qualitative.Paired_12
    # to avoid weird depencency issues installing palettable/colorbrewer
    colors = ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
              '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a' ,'#ffff99', '#b15928')
    return colors

# functions previously defined in a TikZ (LaTeX) image
def hbar2_over_2m():
    return 2.0722  # approimate in meV angstrom^2

def qofef(ef, en, tth):
    from math import sqrt, cos
    return sqrt(ef/hbar2_over_2m()) * sqrt(2 + en/ef - 2 * sqrt(1 + en/ef) * cos(tth))

def qofei(ei, en, tth):
    from numpy import sqrt, cos
    return sqrt(ei/hbar2_over_2m()) * sqrt(2 - en/ei - 2 * sqrt(1 - en/ei) * cos(tth))

def eofqu(ef, q):
    return hbar2_over_2m() * (q * q + 2 * q * sqrt(ef / hbar2_over_2m()))

def eofql(ef, q):
    return hbar2_over_2m() * (q * q - 2 * q * sqrt(ef / hbar2_over_2m()))

def kfsqr(ei, en):
    return (ei - en) / hbar2_over_2m()

def kisqr(ei):
    return ei / hbar2_over_2m()

def kfval(ei, en):
    from math import sqrt
    if en > ei:
        return 0
    return sqrt(kfsqr(ei, en))

def kival(ei):
    from math import sqrt
    return sqrt(kisqr(ei))

def qsqr(ei, en, th):
    from math import cos
    return kisqr(ei) + kfsqr(ei, en) - 2 * kfval(ei, en) * kival(ei) * cos(th)

def qmod(ei, en, th):
    from math import sqrt
    return sqrt(qsqr(ei, en, th))

def vofe(en):
    from math import sqrt
    # sqrt([meV] / [meV angstrom^2]) * [m/s angstrom] -> [m/s]
    return sqrt(en / 81.807) * 3956

def eofv(v):
    # [meV angstrom^2] * ([m/s] / [m/s angstrom])^2 -> [meV]
    return 81.807 * (v / 3956) ** 2

def rep_eis(ei, reps, primary_flight_path, secondary_flight_path, source_freq):
    tau_rep = 1 / source_freq / reps
    v_min = secondary_flight_path / tau_rep
    
    t1 = primary_flight_path / vofe(ei)
    ts = [t1 + n * tau_rep for n in range(reps)]
    vs = [primary_flight_path / t for t in ts]
    es = [eofv(v) for v in vs if v > v_min]
    return es

def rep_enmaxs(ei, reps, primary_flight_path, secondary_flight_path, source_freq):
    tau_rep = 1 / source_freq / reps
    v_min = secondary_flight_path / tau_rep
    ef_min = eofv(v_min)
    
    t1 = primary_flight_path / vofe(ei)
    ts = [t1 + n * tau_rep for n in range(reps)]
    vs = [primary_flight_path / t for t in ts]
    es = [eofv(v) - ef_min for v in vs if v > v_min]
    return es


def q_arc(psi, th0, th1, ei, en):
    from numpy import cos, sin, pi, linspace
    ki, kf = kival(ei), kfval(ei, en)
    th = linspace(th0, th1, 4)/180 * pi
    x = ki - kf * cos(th)
    y = -kf * sin(th)
    cp = cos(psi/180 * pi)
    sp = sin(psi/180 * pi)
    return x * cp - y * sp , y * cp + x * sp

def psi_arc(rx, ry, psi0, psi1):
    from numpy import cos, sin, pi, linspace
    psi = linspace(psi0, psi1, 5)/180 * pi
    x = rx * cos(psi) - ry * sin(psi)
    y = ry * cos(psi) + rx * sin(psi)
    return x, y

def q_xy_path(thmin, thmax, psimin, psimax, ei, en, ki, kf, size, scale, border):
    from math import sqrt, cos, sin
    rf = scale * kf
    # the minimum and maximum radii are just the minimum and maximum accessible Q for given scattering angle
    rn = sqrt(ki * ki + kf * kf - 2 * ki * kf * cos(thmin/180.*pi)) * scale
    rx = sqrt(ki * ki + kf * kf - 2 * ki * kf * cos(thmax/180.*pi)) * scale
    
    if abs(psimax - psimin) >= 360:
        path  = f"M {rx + size + border} {size + border} A {rx} {rx} 0 0 0 {size - rx + border} {size + border} A {rx} {rx} 0 0 0 {rx + size + border} {size + border} Z"
        path += f"M {rn + size + border} {size + border} A {rn} {rn} 0 0 1 {size - rn + border} {size + border} A {rn} {rn} 0 0 1 {rn + size + border} {size + border} Z"
        return path
    
    def coord(x, y):
        return f'{size + x * scale + border} {size - y * scale + border}'
    
    path = "M "
    arc_x, arc_y = q_arc(psimin, thmin, thmax, ei, en)
    path += f" A {rf} {rf} 0 0 0 ".join([coord(x, y) for x, y in zip(arc_x, arc_y)])
    psi_x, psi_y = psi_arc(arc_x[-1], arc_y[-1], 0, psimax-psimin)
    path += " L"
    path += f" A {rx} {rx} 0 0 0 ".join([coord(x, y) for x, y in zip(psi_x, psi_y)])
    arc_x, arc_y = q_arc(psimax, thmax, thmin, ei, en)
    path += " L"
    path += f" A {rf} {rf} 0 0 1 ".join([coord(x, y) for x, y in zip(arc_x, arc_y)])
    psi_x, psi_y = psi_arc(arc_x[-1], arc_y[-1], 0, psimin-psimax)
    path += " L"
    path += f" A {rn} {rn} 0 0 1 ".join([coord(x, y) for x, y in zip(psi_x, psi_y)])
    
    return path


def toward_zero(x):
    from math import floor, ceil
    return ceil(x) if x < 0 else floor(x)

def spec_svg(d: dict):
    return ' '.join([f'{k}="{v}"'.replace('_','-') for k, v in d.items()])

def grid_svg(width, height, size, scale, border, **kwargs):
    group = f"<g {spec_svg(kwargs)}>"
    # We can add lines at integer * scale + size
    lines = ''
    labels = ''
    
    mult = 1
    if toward_zero(size / scale) < 3:
        mult *= 3
        scale /= 3
    if toward_zero(size / scale) > 10:
        mult /= 2
        scale *= 2
        
    for x in range(toward_zero(-size / scale), toward_zero(size / scale)+1):
        p = x * scale + size + border
        lines += f'M {p} {border} l 0 {height} '
        lines += f'M {border} {p} l {width} 0 '
        if x:
            labels += f'<text text-anchor="middle" x="{p}" y="{border+height/2}"><tspan dy="12">{x/mult: 2.1f}</tspan></text>'
            labels += f'<text text-anchor="end" x="{border+width/2}" y="{p}"><tspan dx="-3" dy="5">{x/mult: 2.1f}</tspan></text>'
        
    group += f'<path d="{lines}"/>'
    group += "</g>"
    return group + labels

def label_q_svg(width, height, border, **kwargs):
    spec = spec_svg(kwargs)
    lines = f'<path d="M {border} {height/2 + border} l {width} 0 M {width/2 + border} {border} l 0 {width}" stroke="black"/>'
    x = f'<text x="{width + border-20}" y="{border + height/2}" {spec}><tspan dy="-5">Q<tspan dy="5">x</tspan><tspan dy="-5">&#47;&Aring;</tspan><tspan dy="-5">-1</tspan></tspan></text>'
    y = f'<text x="{width/2 + border}" y="{border-20}" {spec}><tspan dy="15">Q<tspan dy="5">y</tspan><tspan dy="-5">&#47;&Aring;</tspan><tspan dy="-5">-1</tspan></tspan></text>'
    return lines+x+y
    
def max_q_svg(outer, inner, size, border, **kwargs):
    sb = size + border
    path  = f'M {outer + sb} {sb} A {outer} {outer} 0 0 0 {sb - outer} {sb} A {outer} {outer} 0 0 0 {sb + outer} {sb} Z'
    path += f'M {inner + sb} {sb} A {inner} {inner} 0 0 1 {sb - inner} {sb} A {inner} {inner} 0 0 1 {sb + inner} {sb} Z'
    if 'fill' not in kwargs:
        kwargs['fill'] = '#F0F0F0'
    return f'<path d"{path}" {spec_svg(kwargs)}/>'


def qe_trajectory(th, ei, en0, en1):
    from numpy import linspace, pi
    en = linspace(en0, en1, 50)
    x = qofei(ei, en, th / 180 * pi)
    return x, en
    
def qe_path_svg(thmin, thmax, ei, enmin, enmax, xsize, xscale, ysize, yscale, border):
    def coord(x, y):
        return f"{x * xscale + border} {ysize - y * yscale + border}"
    
    path = "M "
    path += f' L '.join([coord(x, y) for x, y in zip(*qe_trajectory(thmin, ei, enmin, enmax))])
    path += " L "
    path += f' L '.join([coord(x, y) for x, y in zip(*qe_trajectory(thmax, ei, enmax, enmin))])
    path += " Z"
    return path

def qe_svg(thmin, thmax, ei, enmin, enmax, xsize, xscale, ysize, yscale, border, **kwargs):
    return f'<path d"{qe_path_svg(thmin, thmax, ei, enmin, enmax, xsize, xscale, ysize, yscale, border)}" {spec_svg(kwargs)}/>'


def list_q_svg(thetas, ei, en, psimin, psimax, width=325, height=325, border=20, **kwargs):
    from math import cos, sin, pi, sqrt
    ki, kf = kival(ei), kfval(ei, en)
    size = min(width, height) / 2 - border
    scale = size / (ki + kfval(ei, 0))
    
    content = f'<svg width="{width}" height="{height}" xmlsn="http://www.w3.org/2000/svg">'
    content += max_q_svg((ki+kf)*scale, (ki-kf)*scale, size, border, fill='#F0F0F0')
    content += grid_svg(width-2*border, height-2*border, size, scale, border, stroke='#aaa')
    content += label_q_svg(width-2*border, height-2*border, border)
    
    spec = spec_svg(kwargs)
    for theta_min, theta_max in thetas:
        path = q_xy_path(theta_min, theta_max, psimin, psimax, ei, en, ki, kf, size, scale, border)
        content += f'<path d="{path}" {spec}/>'
    
    content += '</svg>'
    
    return content

def cspec_ei_reps(ei, reps):
    from math import sqrt
    # 160 m primary flight path, 3.5 m secondary flight path (in plane, plus half of 3.5 m tubes out of plane)
    # 14 Hz source repetition rate
    return rep_eis(ei, reps, 160, 3.5 * sqrt(5) / 2, 14)

def list_q_reps_svg(thetas, ei, en, reps, psimin, psimax, width=325, height=325, border=30, **kwargs):
    from math import cos, sin, pi, sqrt
    ki, kf = kival(ei), kfval(ei, en)
    size = min(width, height) / 2 - border
    scale = size / (ki + kfval(ei, 0))
    
    content = f'<svg width="{width}" height="{height}" xmlsn="http://www.w3.org/2000/svg">'
    content += max_q_svg((ki+kf)*scale, (ki-kf)*scale, size, border, stroke='#F0F0F0', fill='#F0F0F0')
    content += grid_svg(width-2*border, height-2*border, size, scale, border, stroke='#aaa')
    content += label_q_svg(width-2*border, height-2*border, border)
    
    colors = twelve_colors()
    for rep_n, rep_ei in enumerate(cspec_ei_reps(ei, reps)):
        if rep_ei <= en:
            pass
        kwargs['fill'] = colors[rep_n % len(colors)]
        kwargs['stroke'] = colors[rep_n % len(colors)]
        spec = spec_svg(kwargs)
        for theta_min, theta_max in thetas:
            path = q_xy_path(theta_min, theta_max, psimin, psimax, rep_ei, en, kival(rep_ei), kfval(rep_ei, en), size, scale, border)
            content += f'<path d="{path}" {spec}/>'
    
    content += '</svg>'
    return content


def label_e_svg(width, height, border, **kwargs):
    spec = spec_svg(kwargs)
    lines = f'<path d="M {border} {height/2 + border} l {width} 0 M {border} {border} l 0 {width}" stroke="black"/>'
    x = f'<text x="{width + border-20}" y="{border + height/2}" {spec}><tspan dy="-5">Q<tspan dy="5">x</tspan><tspan dy="-5">&#47;&Aring;</tspan><tspan dy="-5">-1</tspan></tspan></text>'
    y = f'<text x="{border}" y="{border}" {spec}><tspan dy="-2" dx="2">E&#47;meV</tspan></text>'
    return lines+x+y

def grid_e_svg(width, height, xsize, xscale, ysize, yscale, border, **kwargs):
    group = f"<g {spec_svg(kwargs)}>"
    # We can add lines at integer * scale + size
    lines = ''
    labels = ''
    
    xmult = 1
    if toward_zero(xsize / xscale) < 3:
        xmult = 3
        xscale /= xmult
    if toward_zero(xsize / xscale) > 10:
        xmult = 1/2
        xscale /= xmult
        
    for x in range(0, toward_zero(xsize / xscale)+1):
        p = x * xscale + border
        lines += f'M {p} {border} l 0 {height} '
        if x:
            labels += f'<text text-anchor="middle" x="{p}" y="{border+height/2}"><tspan dy="12">{x/xmult: 2.1f}</tspan></text>'
        
    ymult = 1
    while toward_zero(ysize / yscale) < 3:
        ymult *= 3
        yscale /= 3
    if toward_zero(ysize / yscale) > 10:
        ymult = 1/2
        yscale /= ymult
    
    for x in range(toward_zero(-ysize / yscale), toward_zero(ysize / yscale)+1):
        p = x * yscale + ysize + border
        neg_p = border + ysize - x * yscale
        lines += f'M {border} {p} l {width} 0 '
        if x:
            labels += f'<text text-anchor="end" x="{border}" y="{neg_p}"><tspan dx="-3" dy="5">{x/ymult: 2.1f}</tspan></text>'
        
    group += f'<path d="{lines}"/>'
    group += "</g>"
    return group + labels

def cspec_en_reps(ei, reps):
    from math import sqrt
    return rep_enmaxs(ei, reps, 160, 3.5 * sqrt(5) / 2, 14)

def list_e_reps_svg(thetas, ei, reps, width=325, height=325, border=30, **kwargs):
    enmin = -1
    xsize = width - 2 * border
    ysize = height / 2 - border
    qscale = xsize / (kival(ei) + kfval(ei, enmin))
    escale = ysize / ei
    
    content  = f'<svg width="{width}" height="{height}" xmlsn="http://www.w3.org/2000/svg">'
    content += qe_svg(0, 180, ei, enmin, ei, xsize, qscale, ysize, escale, border, stroke='#F0F0F0', fill='#F0F0F0')
    content += grid_e_svg(width-2*border, height-2*border, xsize, qscale, ysize, escale, border, stroke='#aaa')
    content += label_e_svg(width-2*border, height-2*border, border)
    
    colors = twelve_colors()
    for rep_n, (rep_ei, rep_en) in enumerate(zip(cspec_ei_reps(ei, reps), cspec_en_reps(ei, reps))):
        if rep_ei <= 0:
            pass
        kwargs['stroke'] = kwargs['fill'] = colors[rep_n % len(colors)]
        spec = spec_svg(kwargs)
        for theta_min, theta_max in thetas:
            path = qe_path_svg(theta_min, theta_max, rep_ei, enmin, rep_en, xsize, qscale, ysize, escale, border)
            content += f'<path d="{path}" {spec}/>'
    
    content += '</svg>'
    return content
    

def magnet_base_path(size, border, outer, inner):
    path  = f"M {outer + size + border} {size + border} A {outer} {outer} 0 0 0 {size - outer + border} {size + border} A {outer} {outer} 0 0 0 {outer + size + border} {size + border} Z"
    path += f"M {inner + size + border} {size + border} A {inner} {inner} 0 0 1 {size - inner + border} {size + border} A {inner} {inner} 0 0 1 {inner + size + border} {size + border} Z"
    return path

def magnet_window_path(size, border, outer, inner, theta_min, theta_max):
    from numpy import cos, sin, pi
    inner_x = inner/size * cos(theta_min/180*pi)
    inner_y = inner/size * sin(theta_min/180*pi)
    outer_x = outer/size * cos(theta_max/180*pi)
    outer_y = outer/size * sin(theta_max/180*pi)
    
    scale = size
    def coord(x, y):
        return f'{size + x * scale + border} {size - y * scale + border}'
    
    path  = "M "
    psi_x, psi_y = psi_arc(inner_x, inner_y, 0, theta_max - theta_min)
    path += f" A {inner} {inner} 0 0 0 ".join([coord(x, y) for x, y in zip(psi_x, psi_y)])
    path += "L "
    psi_x, psi_y = psi_arc(outer_x, outer_y, 0, theta_min - theta_max)
    path += f" A {outer} {outer} 0 0 1 ".join([coord(x, y) for x, y in zip(psi_x, psi_y)])
    path += " Z"
    return path

def magnet_inlet_path(size, border, outer, inner, **kwargs):
    from numpy import arctan2, pi
    width = kwargs.get('width', border)
    theta_min = arctan2(width/2, -outer)/pi*180
    theta_max = (arctan2(-width/2, -outer)/pi*180 - theta_min) % 360 + theta_min  # avoid going the wrong-way-round
    
    path  = f'<path d="{magnet_window_path(size, border, outer, inner, theta_min, theta_max)}" {spec_svg(kwargs)}/>'
    path += f'<path d="M {border} {border + size} l {size - border - inner} 0 l 0 -3 l 9 3 l -9 3 l 0 -3" stroke="black", line-width="4px"/>'
    
    return path

def magnet_svg(thetas, width=325, height=325, border=30, **kwargs):
    size = min(width, height)/2 - border
    inner = kwargs.get('inner', border)
    outer = kwargs.get('outer', size)
    
    content  = f'<svg width="{width}" height="{height}" xmlsn="http://www.w3.org/2000/svg">'
    spec = spec_svg(kwargs)
    content += f'<path d="{magnet_base_path(size, border, outer, inner)}" {spec}/>'
    content += magnet_inlet_path(size, border, outer, inner, fill='#fff', stroke='None')
    kwargs['stroke']='none'
    kwargs['fill']='#fff'
    window_spec = spec_svg(kwargs)
    for theta_min, theta_max in thetas:
        content += f'<path d="{magnet_window_path(size, border, outer, inner, theta_min, theta_max)}" {window_spec}/>'
    
    content += '</svg>'
    return content



# In[88]:

def accessible_QE_tool():
    psi_slider = ipw.FloatRangeSlider(description='$\psi$ / degree', min=-360, max=360, value=[0, 180])
    ei_slider = ipw.FloatSlider(description='$E_i$ / meV', min=0.1, max=25, value=1.5)
    en_slider = ipw.FloatSlider(description='$E$ / meV', min=-1, max=25, value=0)
    e_link = ipw.jslink((ei_slider, 'value'), (en_slider, 'max'))
    rep_slider = ipw.IntSlider(description='# $E_i$', min=1, max=15, value=1)
    
    ei_label = ipw.Label(value='$E_i =$')
    ei_out = ipw.Label()
    q_out = ipw.HTML() 
    e_out = ipw.HTML()
    m_out = ipw.HTML()
    box = ipw.HBox()
    
    def get_controls(b):
        h_qe_plots, h_controls_magnet = b.children[0].children
        h_controls = h_controls_magnet.children[0]
        return h_controls
    
    def remove_detector_coverage(owner, mn, mx):
        def hbox_matches(hbox):
            if isinstance(hbox, ipw.HBox) and len(hbox.children) == 3:
                one, two, three = hbox.children
                return one == owner and two == mn and three == mx
            return False
        
        left = get_controls(box)
        left.children = tuple([l for l in left.children if not hbox_matches(l)])
        update_output('new')
    
    def add_detector_coverage_range(owner, min=0, max=0):
        bt = ClickToggleButtons(options=[''], tooltip='Remove range', icons=['remove'], style={'button_width': '5px'})
        mn = ipw.BoundedFloatText(min=-180, max=180, step=1, value=min,
                                  description='$\\theta_\\text{min}$',
                                  style={'description_width': 'auto'},
                                  layout = ipw.Layout(width='100px'))
        mx = ipw.BoundedFloatText(min=-180, max=180, step=1, value=max,
                                  description='$\\theta_\\text{max}$',
                                  style={'description_width': 'auto'},
                                  layout = ipw.Layout(width='100px'))
        left = get_controls(box)
        left.children = tuple(*[list(left.children) + [ipw.HBox([bt, mn, mx])]])
        bt.on_click(lambda x: remove_detector_coverage(x, mn, mx))
        mn.observe(update_output, 'value')
        mx.observe(update_output, 'value')
        
        
    def update_output(args):
        def is_theta(hbox):
            if isinstance(hbox, ipw.HBox) and len(hbox.children) == 3:
                return True
            return False
        
        def theta_values(hbox):
            one, two, three = hbox.children
            return two.value, three.value
        
        left = get_controls(box)
        thetas = [theta_values(l) for l in left.children if is_theta(l)]
        
        ei = ei_slider.value
        en = en_slider.value
        psi_min, psi_max = psi_slider.value
        reps = rep_slider.value
        
        q_out.value = list_q_reps_svg(thetas, ei, en, reps, psi_min, psi_max, width=500, height=500, fill_opacity=0.5)  
        e_out.value = list_e_reps_svg(thetas, ei, reps, width=500, height=500, fill_opacity=0.5)
        m_out.value = magnet_svg(thetas, width=400, height=400, stroke='#0099dc', fill='#0099dc')
        
        eis = ', '.join([f'{e:0.2f}' for e in cspec_ei_reps(ei_slider.value, rep_slider.value)])
        ei_out.value = fr'{eis} meV'
        
    psi_slider.observe(update_output, 'value')
    ei_slider.observe(update_output, 'value')
    en_slider.observe(update_output, 'value')
    rep_slider.observe(update_output, 'value')
    
    add_coverage_button = ipw.Button(description='Add detector range')
    add_coverage_button.on_click(add_detector_coverage_range)
    
    controls = ipw.VBox([ei_slider, en_slider, psi_slider, rep_slider, add_coverage_button])
    qe_plots = ipw.VBox([ipw.HBox([ei_label, ei_out]), ipw.HBox([q_out, e_out])])
    controls_and_magnet = ipw.HBox([controls, m_out])
    box.children = [ipw.VBox([qe_plots, controls_and_magnet])]
    
    for n, x in (-20, -5), (5,20), (25,40), (45,60), (65,80), (85,100), (105,120), (125, 140):
        add_detector_coverage_range(None, n, x)
        
    update_output('')

    return box

