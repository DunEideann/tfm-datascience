# GRAFICOS de 10 PREDICCIONES
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/similarity'
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')

from PIL import Image
def imageCombiner(images_path = [], extra='', extension='png', direction='horizontal'):
    images = [Image.open(x) for x in images_path]
    size = [image.size for image in images]

    # Calcular la altura total como la mayor altura entre todas las imágenes
    total_height = max([height for (width, height) in size])

    # Calcular el ancho total como la suma de los anchos de todas las imágenes
    total_width = sum([width for (width, height) in size])

    # Crear una nueva imagen con el tamaño combinado
    new_image = Image.new('RGBA', (total_width, total_height))

    # Pegar las imágenes una al lado de la otra (de izquierda a derecha)
    acumulated_width = 0
    for image in images:
        new_image.paste(image, (acumulated_width, 0))
        acumulated_width += image.size[0]

    # Guardar la imagen resultante
    new_image.save(f'{FIGS_PATH}/variance_{extra}.{extension}')

ccsignal_periods = ['MEDIUM', 'LONG']
statistics = ['1Quantile', '99Quantile', 'Mean']
numbers = [5, 6]
periods = ['test', f'{future_4[0]}-{future_4[1]}', f'{future_3[0]}-{future_3[1]}']


for statistic in statistics:
    for predictands_num in numbers:
        images_cc = []
        images_ssp = []
        images_mean_sd_cc = []
        images_mean_sd = []
        for period in periods:
            opt = 'ssp585' if period!='test' else ''
            opt2 = '_' if period!='test' else ''
            image_name_ssp = f'{FIGS_PATH}/variancesGraph_{opt}{period}{opt2}{statistic}_Percentage_{predictands_num}.png'
            images_ssp.append(image_name_ssp)
            image_mean_sd =  f'{FIGS_PATH}/meanSd_{opt}{period}{opt2}{statistic}_Percentage_{predictands_num}.png'
            images_mean_sd.append(image_mean_sd)

        for cc_period in ccsignal_periods:
            image_name_cc = f'{FIGS_PATH}/variancesGraph_ccsignal{statistic}_Percentage_{predictands_num}_{cc_period}.png'
            images_cc.append(image_name_cc)
            image_mean_sd_cc =  f'{FIGS_PATH}/meanSd_ccsignal{statistic}_Percentage_{predictands_num}_{cc_period}.png'
            images_mean_sd_cc.append(image_mean_sd_cc)


        imageCombiner(images_path=images_ssp, extra=f'prediction_{statistic}_{predictands_num}')
        imageCombiner(images_path=images_cc, extra=f'ccsignal_{statistic}_{predictands_num}')
        imageCombiner(images_path=images_mean_sd, extra=f'mean_sd_{statistic}_{predictands_num}')
        imageCombiner(images_path=images_mean_sd_cc, extra=f'mean_sd_ccsignal_{statistic}_{predictands_num}')
