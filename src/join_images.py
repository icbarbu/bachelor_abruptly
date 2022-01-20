from PIL import Image
import glob
import sys

dir = 'experiments/anal/'

pres =  ['h_foraging-TD']
post = range(1, 21+1)

for pre in pres:
    for pos in post:
        type = pre+'_'+str(pos)+'_'
        print(type)

        matching_files = glob.glob(f'{dir}*{type}*.*')
        matching_files.sort()
        images = [Image.open(x) for x in matching_files]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height), 'white')

        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        new_im.save(f'{dir}{type}.png')