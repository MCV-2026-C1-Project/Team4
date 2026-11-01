"""
C1 - Final presentation: builds Team4_final_presentation_v1.pptx from the figures in ../results/figures.

Needs python-pptx (only for the slides, so it is not in requirements.txt). Run from this folder:
    python build_deck.py
"""

import copy
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_THEME_COLOR as TC
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'results' / 'figures'
OUT = HERE / 'Team4_final_presentation_v1.pptx'
FOOTER = 'Team 4  ·  Martí  ·  Li Boquian  ·  Maren  ·  Max Mira'
W, H = 13.333, 7.5
LEFT = 0.6                      # left margin of every slide
TOP = 1.55                      # first y under the title
GREY = RGBColor(0x8C, 0x96, 0x92)   # week 3 in the charts, hex because chart series need a fixed value
THEME = dict(dk1='1E2B28', lt1='FFFFFF', dk2='16302A', lt2='EEF2F0', accent1='2F6F5E', accent2='C8553D', accent3='C8912E',
             accent4='8C9692', accent5='5B8F7F', accent6='7A4A2E', hlink='2F6F5E', folHlink='7A4A2E')


# ----------------------------------------------------------------------------- theme and layouts
def set_theme(prs):
    theme = next(r.target_part for r in prs.slide_master.part.rels.values() if r.reltype.endswith('/theme'))
    root = etree.fromstring(theme.blob); ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    root.set('name', 'Team4 gallery')
    scheme = root.find('.//a:clrScheme', ns); scheme.set('name', 'Team4 gallery')
    for name, value in THEME.items():
        slot = scheme.find(f'a:{name}', ns)
        for child in list(slot): slot.remove(child)
        etree.SubElement(slot, qn('a:srgbClr')).set('val', value)
    fonts = root.find('.//a:fontScheme', ns); fonts.set('name', 'Team4 gallery')
    fonts.find('a:majorFont/a:latin', ns).set('typeface', 'Cambria')
    fonts.find('a:minorFont/a:latin', ns).set('typeface', 'Calibri')
    theme._blob = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


def drop_layouts(prs, keep):
    layouts = prs.slide_layouts
    for layout in list(layouts):
        if layout.name not in keep: layouts.remove(layout)


def footer_field(layout, dark=False):
    """Team names and page number live on the layout, so they appear on every slide."""
    colour = 'bg2' if dark else 'accent4'
    run = f'<a:rPr lang="en-GB" sz="1100"><a:solidFill><a:schemeClr val="{colour}"/></a:solidFill></a:rPr>'
    def box(sid, name, x, w, para):
        return (f'<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f'<p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr userDrawn="1"/></p:nvSpPr>'
                f'<p:spPr><a:xfrm><a:off x="{Inches(x)}" y="{Inches(6.98)}"/><a:ext cx="{Inches(w)}" cy="{Inches(0.3)}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
                f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/><a:lstStyle/>{para}</p:txBody></p:sp>')
    tree = layout.shapes._spTree
    tree.append(etree.fromstring(box(101, 'Team footer', LEFT, 8, f'<a:p><a:r>{run}<a:t>{FOOTER}</a:t></a:r></a:p>')))
    number = f'<a:p><a:pPr algn="r"/><a:fld id="{{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}}" type="slidenum">{run}<a:t>‹#›</a:t></a:fld></a:p>'
    tree.append(etree.fromstring(box(102, 'Page number', W - LEFT - 1, 1, number)))


def strip_placeholders(layout, keep=('title',)):
    for sh in list(layout.placeholders):
        kind = str(sh.placeholder_format.type).lower()
        if not any(k in kind for k in keep): sh._element.getparent().remove(sh._element)


def set_background(obj, theme_colour):
    fill = obj.background.fill; fill.solid(); fill.fore_color.theme_color = theme_colour


def style_title(ph, x, y, w, h, size, colour):
    ph.left, ph.top, ph.width, ph.height = Inches(x), Inches(y), Inches(w), Inches(h)
    tf = ph.text_frame; tf.vertical_anchor = MSO_ANCHOR.TOP; no_margin(tf); tf.word_wrap = True
    body = tf._txBody.find(qn('a:bodyPr'))
    for c in list(body): body.remove(c)
    etree.SubElement(body, qn('a:noAutofit'))
    lst = tf._txBody.find(qn('a:lstStyle'))
    for c in list(lst): lst.remove(c)
    lvl = etree.SubElement(lst, qn('a:lvl1pPr')); lvl.set('algn', 'l')
    rp = etree.SubElement(lvl, qn('a:defRPr')); rp.set('sz', str(size * 100)); rp.set('b', '1')
    fill = etree.SubElement(rp, qn('a:solidFill')); etree.SubElement(fill, qn('a:schemeClr')).set('val', colour)


def build_layouts(prs):
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    set_theme(prs)
    drop_layouts(prs, {'Title Slide', 'Title Only', 'Section Header'})
    by = {l.name: l for l in prs.slide_layouts}
    for ph in list(prs.slide_master.placeholders):
        kind = str(ph.placeholder_format.type).lower()
        if any(k in kind for k in ('date', 'footer', 'slide_number')): ph._element.getparent().remove(ph._element)
    content, dark_title, closing = by['Title Only'], by['Title Slide'], by['Section Header']
    content.name, dark_title.name, closing.name = 'Content', 'Opening', 'Closing'
    strip_placeholders(content); style_title(content.placeholders[0], LEFT, 0.5, W - 2 * LEFT, 0.8, 36, 'tx2')
    footer_field(content)
    strip_placeholders(dark_title, ('title', 'subtitle'))
    style_title(dark_title.placeholders[0], LEFT, 1.6, 7.9, 2.6, 48, 'bg1')
    sub = [p for p in dark_title.placeholders if p.placeholder_format.idx != 0][0]
    sub.left, sub.top, sub.width, sub.height = Inches(LEFT), Inches(4.35), Inches(7.9), Inches(1.1)
    set_background(dark_title, TC.TEXT_2); footer_field(dark_title, dark=True)
    strip_placeholders(closing); style_title(closing.placeholders[0], LEFT, 0.5, W - 2 * LEFT, 0.8, 36, 'bg1')
    set_background(closing, TC.TEXT_2); footer_field(closing, dark=True)
    return content, dark_title, closing


# ----------------------------------------------------------------------------- helpers
def no_margin(tf):
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0


def set_text(tf, text, size, colour=TC.TEXT_1, bold=False, align=None, font=None):
    no_margin(tf); tf.word_wrap = True
    p = tf.paragraphs[0]; add_run(p, text, size, colour, bold, font)
    if align: p.alignment = align
    return p


def add_run(p, text, size, colour=TC.TEXT_1, bold=False, font=None):
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.theme_color = colour
    if font: r.font.name = font
    return r


def text(slide, x, y, w, h, value, size=16, colour=TC.TEXT_1, bold=False, name=None, align=None, font=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); box.name = name or value[:30]
    set_text(box.text_frame, value, size, colour, bold, align, font)
    return box


def bullets(slide, x, y, w, h, items, size=16, colour=TC.TEXT_1, gap=10, name='Key points'):
    """items: strings, or (bold lead, rest) tuples."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); box.name = name
    tf = box.text_frame; no_margin(tf); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        ppr = p._p.get_or_add_pPr(); ppr.set('marL', '228600'); ppr.set('indent', '-228600')
        etree.SubElement(ppr, qn('a:buFont')).set('typeface', 'Arial'); etree.SubElement(ppr, qn('a:buChar')).set('char', '•')
        lead, rest = item if isinstance(item, tuple) else (None, item)
        if lead: add_run(p, lead + ' ', size, colour, True)
        add_run(p, rest, size, colour)
    return box


def card(slide, x, y, w, h, fill=TC.BACKGROUND_2, name='Card'):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.name = name
    s.adjustments[0] = 0.05; s.fill.solid(); s.fill.fore_color.theme_color = fill; s.line.fill.background(); s.shadow.inherit = False
    return s


def framed(slide, file, x, y, w, h, alt, mat=0.07):
    """The motif of the deck: every picture hangs in a thin dark frame with a white mat, centred in the box."""
    iw, ih = Image.open(FIG / file).size; scale = min((w - 2 * mat) / iw, (h - 2 * mat) / ih); pw, ph = iw * scale, ih * scale
    fw, fh = pw + 2 * mat, ph + 2 * mat; fx, fy = x + (w - fw) / 2, y + (h - fh) / 2
    frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(fx), Inches(fy), Inches(fw), Inches(fh)); frame.name = f'Frame {file}'
    frame.fill.solid(); frame.fill.fore_color.theme_color = TC.BACKGROUND_1; frame.shadow.inherit = False
    frame.line.color.theme_color = TC.TEXT_2; frame.line.width = Pt(1.5)
    pic = slide.shapes.add_picture(str(FIG / file), Inches(fx + mat), Inches(fy + mat), Inches(pw), Inches(ph)); pic.name = file
    pic._element.nvPicPr.cNvPr.set('descr', alt)
    return fx, fy, fw, fh


def stat(slide, x, y, w, value, label, colour=TC.ACCENT_1):
    text(slide, x, y, w, 0.95, value, 54, colour, True, f'Stat {value}', font='Cambria')
    text(slide, x, y + 0.98, w, 0.6, label, 14, TC.TEXT_1, name=f'Label {value}')


def bar_chart(slide, x, y, w, h, cats, series, kind, colours, number='0.000', vmax=1.0, title=None, legend=True, name='Chart', alt=''):
    data = CategoryChartData(number_format=number); data.categories = cats
    for n, v in series: data.add_series(n, v)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data); gf.name = name
    gf._element.nvGraphicFramePr.cNvPr.set('descr', alt)
    ch = gf.chart; ch.font.size = Pt(14); ch.font.color.theme_color = TC.TEXT_1
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title; run = ch.chart_title.text_frame.paragraphs[0].runs[0]; run.font.size = Pt(14); run.font.bold = True
        ch.chart_title.include_in_layout = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(14)
    va = ch.value_axis; va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0; va.maximum_scale = vmax
    ca = ch.category_axis; ca.format.line.color.theme_color = TC.ACCENT_4; ca.tick_labels.font.size = Pt(14); ca.has_major_gridlines = False
    plot = ch.plots[0]; plot.gap_width = 60; plot.overlap = -5; plot.has_data_labels = True
    dl = plot.data_labels; dl.show_value = True; dl.number_format = number; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(14)
    for s, c in zip(plot.series, colours):
        s.format.fill.solid()
        if isinstance(c, RGBColor): s.format.fill.fore_color.rgb = c
        else: s.format.fill.fore_color.theme_color = c
        for i, v in enumerate(s.values):        # fixed text: a number format would follow the regional decimal comma
            label = s.points[i].data_label; label.position = XL_LABEL_POSITION.OUTSIDE_END
            label.text_frame.text = f"{v:.{len(number.split('.')[1])}f}"
            label.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    return ch


def caption(slide, y, value, x=LEFT, w=W - 2 * LEFT):
    text(slide, x, y, w, 0.3, value, 12, TC.ACCENT_4, name='Caption')


# ----------------------------------------------------------------------------- slides
def s_title(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Finding museum paintings by their keypoints'
    sub = [p for p in s.placeholders if p.placeholder_format.idx != 0][0]
    tf = sub.text_frame; no_margin(tf)
    add_run(tf.paragraphs[0], 'C1 project  ·  Final presentation  ·  Team 4', 20, TC.BACKGROUND_1, True)
    p = tf.add_paragraph(); add_run(p, 'Martí  ·  Li Boquian  ·  Maren  ·  Max Mira', 18, TC.BACKGROUND_2)
    for para in tf.paragraphs:
        para.alignment = PP_ALIGN.LEFT
        para._p.get_or_add_pPr().set('marL', '0'); para._p.get_or_add_pPr().set('indent', '0')
        etree.SubElement(para._p.get_or_add_pPr(), qn('a:buNone'))
    framed(s, 'matches_known.png', 9.15, 1.6, 3.6, 4.3, 'A query photo and its BBDD painting joined by green inlier matches')
    s.notes_slide.notes_text_frame.text = (
        'Team 4. We focus on week 4, local features, and show how it builds on the previous weeks. Each member presents one part.')


def s_result(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Keypoints beat texture on the harder query set'
    stat(s, LEFT, TOP, 3.9, '0.974', 'mAP@1 and mAP@5 on QSD1-W4\nWeek 3 on the same set: 0.923')
    stat(s, LEFT, TOP + 1.75, 3.9, '38 / 39', 'entries right: 29 paintings + 10 unknown photos, one painting lost')
    stat(s, LEFT, TOP + 3.5, 3.9, '10 / 10', 'unknown photos discarded\nF1 0.952 (P 0.909, R 1.000)')
    bar_chart(s, 4.9, TOP - 0.05, W - 4.9 - LEFT, 4.95, ['Without discard', 'With discard'],
              [('Week 3: texture fusion', (0.692, 0.923)), ('Week 4: SIFT + RANSAC', (0.718, 0.974))], XL_CHART_TYPE.COLUMN_CLUSTERED,
              [GREY, TC.ACCENT_1], title='mAP@1 on QSD1-W4', name='mAP@1 week 3 vs week 4',
              alt='Without discard: week 3 0.692, week 4 0.718. With discard: week 3 0.923, week 4 0.974.')
    caption(s, 6.55, 'Week 3 has no discard of its own: we added a threshold optimised with the same F1 criterion.', x=4.9, w=W - 4.9 - LEFT)
    s.notes_slide.notes_text_frame.text = (
        'The result first. On QSD1-W4 we reach 0.974, with mAP@1 equal to mAP@5, against 0.923 for our best week-3 system. '
        'That is 38 of 39 entries against 36, so the gain on this small set is modest. Handling unknown queries is what moves the score most: '
        'both systems are near 0.7 without it. Later we show that the difference is much clearer on independent data.')


def s_journey(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'From global histograms to verified local matches'
    weeks = [('Week 1', '0.900', '1D colour histograms', 'HSV and CIELab, illumination normalised, chi-square and Hellinger'),
             ('Week 2', '0.967', '3D histograms + background', 'YCrCb spatial pyramid. Wall model removes the background'),
             ('Week 3', '1.000', 'Texture + two paintings', 'Median denoising. Fusion of LBP, DCT, wavelet and Gabor'),
             ('Week 4', '0.974', 'Keypoints + unknowns', 'SIFT, ratio test, RANSAC. Queries below the inlier threshold are discarded')]
    gap, cw = 0.3, (W - 2 * LEFT - 3 * 0.3) / 4
    for i, (wk, score, head, body) in enumerate(weeks):
        x = LEFT + i * (cw + gap); last = i == 3; ink = TC.BACKGROUND_1 if last else TC.TEXT_1
        card(s, x, TOP, cw, 3.95, TC.ACCENT_1 if last else TC.BACKGROUND_2, f'Card {wk}')
        text(s, x + 0.22, TOP + 0.2, cw - 0.44, 0.35, wk, 16, ink, True, f'{wk} label')
        text(s, x + 0.22, TOP + 0.55, cw - 0.44, 0.85, score, 44, ink, True, f'{wk} score', font='Cambria')
        text(s, x + 0.22, TOP + 1.4, cw - 0.44, 0.3, 'mAP@1', 12, ink, name=f'{wk} metric')
        text(s, x + 0.22, TOP + 1.8, cw - 0.44, 0.7, head, 16, ink, True, f'{wk} heading')
        text(s, x + 0.22, TOP + 2.5, cw - 0.44, 1.3, body, 14, ink, name=f'{wk} body')
    text(s, LEFT, 5.75, W - 2 * LEFT, 0.5, 'Week 4 still uses the week-3 denoising and segmentation.', 18, TC.TEXT_2, True, 'Link between weeks')
    caption(s, 6.3, 'Development scores. Each week has its own query set, so the values are not directly comparable. Week 2 is QSD1 (QSD2: 1.000).')
    s.notes_slide.notes_text_frame.text = (
        'The global scheme is always query by example: describe, compare, rank. Each week changed the descriptor and made the queries harder. '
        'Week 3 reached 1.000 on its development sets, so those sets no longer separated methods. Week 4 adds non-cropped photos with up to two paintings, '
        'colour changes, noise, and queries that are not in the database. Do not read 1.000 to 0.974 as a loss: the set is harder.')


def s_pipeline(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Every match must agree on one homography'
    steps = [('Denoise', 'median 3×3 twice, only if impulses are found'), ('Keypoints', 'DoG + SIFT, up to 2000, longest side 800 px'),
             ('Tentative matches', 'nearest neighbours, Lowe ratio 0.75, one to one'), ('Verify', 'RANSAC homography, implausible models rejected'),
             ('Score', 'inliers per BBDD image: rank them, or return [[-1]] below T = 25')]
    y = TOP
    for i, (head, body) in enumerate(steps):
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(LEFT), Inches(y + 0.03), Inches(0.5), Inches(0.5)); dot.name = f'Step {i + 1}'
        dot.fill.solid(); dot.fill.fore_color.theme_color = TC.ACCENT_1; dot.line.fill.background(); dot.shadow.inherit = False
        tf = dot.text_frame; no_margin(tf); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_text(tf, str(i + 1), 16, TC.BACKGROUND_1, True, PP_ALIGN.CENTER)
        text(s, LEFT + 0.72, y - 0.03, 5.6, 0.32, head, 16, TC.TEXT_2, True, f'{head} heading')
        text(s, LEFT + 0.72, y + 0.28, 5.6, 0.55, body, 14, name=f'{head} detail')
        y += 1.0
    fx, fy, fw, fh = framed(s, 'matches_known.png', 7.2, TOP, W - 7.2 - LEFT, 4.85, 'A query photo on the left and its BBDD painting on the right, joined by green inlier matches')
    caption(s, 6.5, 'Image 4 with BBDD 161: 1163 inliers.', x=7.2, w=W - 7.2 - LEFT)
    s.notes_slide.notes_text_frame.text = (
        'Task 1 and Task 2. We tested three options: SIFT, RootSIFT and ORB. Appearance alone gives tentative matches; the geometry decides. '
        'A flat painting seen from another point of view must follow one homography, so wrong matches rarely agree. '
        'Cross-check gave no improvement and took longer, so it is not used. The search is exhaustive and the seed fixed, so results repeat exactly. '
        'The BBDD features are cached and a test run takes about one minute.')


def s_two(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Segmentation only fixes the layout'
    framed(s, 'matches_two.png', LEFT, TOP, W - 2 * LEFT, 2.75, 'A photo with two paintings joined to their two BBDD images, green lines for the first painting and orange for the second')
    bar_chart(s, LEFT, 4.4, 5.0, 2.45, ['Week-3 segments', 'Whole photo', 'Hybrid'], [('mAP@1', (0.897, 0.923, 0.974))], XL_CHART_TYPE.BAR_CLUSTERED,
              [TC.ACCENT_1], legend=False, title='SIFT, mAP@1 with discard', name='mAP@1 per strategy',
              alt='Week-3 segments 0.897, whole photo 0.923, hybrid 0.974.').category_axis.reverse_order = True
    bullets(s, 5.95, 4.45, W - 5.95 - LEFT, 2.4, [
        ('Segments:', 'a white mat looks like the wall, so image 28 loses a painting and both slots.'),
        ('Whole photo:', 'sequential RANSAC finds both, but a lone match can land in the wrong slot (image 16).'),
        ('Hybrid:', 'match the whole photo, and use segments only to place what is missing.')], size=16, gap=6)
    s.notes_slide.notes_text_frame.text = (
        'Up to two paintings per photo. Matching the whole photo, then removing the inliers of the first painting, finds both paintings without any segmentation. '
        'The metric is positional: a painting in the wrong slot loses both entries. So we keep the week-3 segmentation only for the layout. '
        'This is the single change that moved us most, from 0.923 to 0.974.')


def s_unknowns(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Inlier count separates unknown queries'
    framed(s, 'scores.png', LEFT, TOP, 8.0, 2.75, 'Dot plot of best inliers per query on a log scale: unknown queries at 7 or fewer, known queries at 53 or more except one at 5, with the threshold at 25')
    framed(s, 'matches_unknown.png', LEFT, 4.45, 8.0, 2.4, 'An unknown query of slate-like texture against a BBDD painting, joined by only four unrelated matches')
    bullets(s, 9.0, TOP, W - 9.0 - LEFT, 5.2, [
        ('Unknown:', '7 inliers at most.'),
        ('Known:', '53 or more, except painting 128.'),
        ('Threshold:', 'every T in (7, 53] gives the best F1 on QSD1-W4, 0.952.'),
        ('T = 25:', 'the best of that range on independent data.'),
        ('Bottom image:', 'best match of an unknown query, 4 inliers with no geometric sense.')], size=16, gap=12)
    s.notes_slide.notes_text_frame.text = (
        'Task 2, part two. A query is unknown when its best score is below T. We optimise F1 with unknown as the positive class. '
        'All 10 unknowns are discarded. The single false alarm is painting 128, a known one, which gives P 0.909 and R 1.000. '
        'F1 alone does not fix T, because a wide range ties. The next slide shows how we chose inside it.')


def s_choice(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'QSD1-W4 ties, independent data picks SIFT'
    text(s, LEFT, TOP, 4.6, 0.35, 'QSD1-W4 cannot decide', 18, TC.TEXT_2, True, 'Left heading')
    bullets(s, LEFT, TOP + 0.45, 4.6, 2.0, [
        'SIFT, RootSIFT and ORB all give mAP 0.974 and leave-one-out F1 0.952.',
        'ORB reaches F1 1.000 through one photo: 9 inliers against 7.'], size=16, gap=8, name='Why the set ties')
    text(s, LEFT, TOP + 2.5, 4.6, 0.35, 'Our check', 18, TC.TEXT_2, True, 'Check heading')
    bullets(s, LEFT, TOP + 2.95, 4.6, 2.2, [
        '60 photos from QSD1-W3 and QSD2-W3.',
        'Known: the full BBDD. Unknown: the right paintings removed.',
        'SIFT vs RootSIFT is one photo: weak evidence.'], size=16, gap=8, name='How we checked')
    bar_chart(s, 5.6, TOP - 0.05, W - 5.6 - LEFT, 4.9, ['SIFT (T 25)', 'RootSIFT (T 25)', 'ORB (T 8)', 'Week 3 (T 8.94)'],
              [('Known missed', (3.3, 3.3, 1.7, 30.0)), ('False known', (0.0, 1.7, 8.3, 6.7))], XL_CHART_TYPE.COLUMN_CLUSTERED,
              [TC.ACCENT_1, TC.ACCENT_2], number='0.0', vmax=35, title='Error rate on the week-3 sets (%)', name='Independent check',
              alt='Known missed and false known in percent. SIFT 3.3 and 0. RootSIFT 3.3 and 1.7. ORB 1.7 and 8.3. Week 3 30 and 6.7.')
    caption(s, 6.5, 'BBDD near-duplicates (179 / 219, 13 / 52) explain the top simulated-unknown scores.', x=5.6, w=W - 5.6 - LEFT)
    s.notes_slide.notes_text_frame.text = (
        'On the development set the three descriptors tie, so we needed another criterion. The week-3 sets have ground truth and were not used to design the week-4 pipeline, although week 3 itself was developed on them. '
        'Removing the right painting from the database simulates an unknown query. SIFT is the only option with no false known at T = 25. '
        'ORB finds more paintings but accepts more wrong ones. The texture system of week 3 misses 30 percent of the known queries at its threshold. '
        'This is the clearest evidence in favour of keypoints.')


def s_limits(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Flat, reflective paintings remain out of reach'
    pw = 1.72; h = 2.6
    pairs = [('fail_128_query.png', 'Query, image 8', 'A dark painting covered with plastic, photographed in the museum'),
             ('fail_128_bbdd.png', 'BBDD 128', 'The same plastic-covered painting in the BBDD, with a reflection of the photographer'),
             ('fail_11_query.png', 'Query, image 16', 'A nearly uniform blue painting, as it appears in the photo'),
             ('fail_11_bbdd.png', 'BBDD 11', 'The same painting in the BBDD, where a fine surface texture is visible')]
    for i, (file, label, alt) in enumerate(pairs):
        col, row = i % 2, i // 2; x = LEFT + col * (pw + 0.25); y = TOP + row * (h + 0.05)
        framed(s, file, x, y, pw, 2.2, alt)
        text(s, x, y + 2.24, pw, 0.28, label, 12, TC.ACCENT_4, name=f'Label {label}', align=PP_ALIGN.CENTER)
    x = 4.7; w = W - x - LEFT
    text(s, x, TOP, w, 0.35, 'What fails', 18, TC.TEXT_2, True, 'Failures heading')
    bullets(s, x, TOP + 0.45, w, 2.1, [
        ('Painting 128:', 'reflections change between photos. Only 5 inliers, so it is discarded as unknown.'),
        ('Painting 11:', 'its fine texture vanishes. Right with 7 inliers, below T, only thanks to the layout rule: 0.026 of mAP.')], size=16, gap=8, name='What fails')
    text(s, x, TOP + 2.7, w, 0.35, 'What could fail on the test', 18, TC.TEXT_2, True, 'Risks heading')
    bullets(s, x, TOP + 3.15, w, 2.1, [
        ('Mixed photos:', 'a known and an unknown painting together give a ranking, not [-1]. QSD1-W4 has none.'),
        ('Small sample:', '10 unknown photos, 4 of them repeating 2 paintings.')], size=16, gap=8, name='Risks')
    s.notes_slide.notes_text_frame.text = (
        'Being honest about the limits. Painting 128 has plastic with changing reflections, and the database image even shows the photographer reflected. '
        'Painting 11 has almost no structure at this scale. Local features need local structure, so these cases fail by nature. '
        'We expect the test score to be lower than the development score.')


def s_close(prs, layout):
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = 'Verification is what made retrieval reliable'
    items = [('Geometry over appearance', 'Inliers put known and unknown queries far apart: 53 or more against 7 or fewer, with one exception.'),
             ('Earlier work still counts', 'Week-3 denoising and segmentation stay in the pipeline, used only where they help.'),
             ('Do not trust a tie', 'When a small set cannot separate methods, independent data can: it chose both the descriptor and T.')]
    cw = (W - 2 * LEFT - 2 * 0.3) / 3
    for i, (head, body) in enumerate(items):
        x = LEFT + i * (cw + 0.3)
        card(s, x, TOP + 0.45, cw, 3.7, TC.BACKGROUND_1, f'Card {head}')
        text(s, x + 0.28, TOP + 0.7, cw - 0.56, 0.9, head, 24, TC.TEXT_2, True, f'{head} heading', font='Cambria')
        text(s, x + 0.28, TOP + 1.75, cw - 0.56, 2.2, body, 18, name=f'{head} body')
    s.notes_slide.notes_text_frame.text = (
        'Three takeaways. Thank you, we are happy to take questions.')


def main():
    prs = Presentation()
    content, opening, closing = build_layouts(prs)
    s_title(prs, opening); s_result(prs, content); s_journey(prs, content); s_pipeline(prs, content); s_two(prs, content)
    s_unknowns(prs, content); s_choice(prs, content); s_limits(prs, content); s_close(prs, closing)
    prs.core_properties.title = 'Finding museum paintings by their keypoints'; prs.core_properties.author = 'Team 4'
    prs.save(OUT); print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
