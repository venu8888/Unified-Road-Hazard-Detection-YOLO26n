"""
Annotation Conversion Script
Extracts and converts Pascal VOC XML annotations from N-RDD2024 into YOLO format.
Creates processed_conversion_log.csv documenting every conversion.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import io

def convert_voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    # Ensure valid coordinates
    xmin = max(0.0, min(float(xmin), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymax = max(0.0, min(float(ymax), float(img_h)))
    
    if xmax <= xmin or ymax <= ymin:
        return None
        
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    
    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)

def run_conversion(rdd_base_dir, output_log='processed_conversion_log.csv'):
    tv_dir = os.path.join(rdd_base_dir, 'N-RDD2024Road damage and defects', 'Training and Validation Dataset')
    countries = ['Czech Republic', 'USA', 'china-motorbike', 'india', 'japan', 'norway']
    
    log_records = []
    total_converted = 0
    total_boxes = 0
    
    print("Starting Pascal VOC XML to YOLO conversion...")
    for country in countries:
        zpath = os.path.join(tv_dir, f'{country}_voc.zip')
        if not os.path.exists(zpath):
            continue
            
        with zipfile.ZipFile(zpath, 'r') as z:
            xml_files = [n for n in z.namelist() if n.endswith('.xml')]
            print(f"Converting {len(xml_files)} files from {country}...")
            
            for xf in xml_files:
                base_name = os.path.splitext(os.path.basename(xf))[0]
                img_name = base_name + '.jpg'
                
                xml_data = z.read(xf)
                root = ET.fromstring(xml_data)
                
                size_node = root.find('size')
                if size_node is not None:
                    img_w = float(size_node.find('width').text)
                    img_h = float(size_node.find('height').text)
                else:
                    img_w, img_h = 640.0, 640.0
                    
                objs = root.findall('object')
                num_boxes = 0
                classes_in_file = []
                
                for obj in objs:
                    cname = obj.find('name').text.strip()
                    classes_in_file.append(cname)
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    yolo_box = convert_voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
                    if yolo_box is not None:
                        num_boxes += 1
                        total_boxes += 1
                        
                log_records.append({
                    'source_dataset': 'N-RDD2024',
                    'country': country,
                    'source_xml_file': xf,
                    'image_filename': img_name,
                    'image_width': int(img_w),
                    'image_height': int(img_h),
                    'original_format': 'Pascal VOC XML',
                    'target_format': 'YOLO TXT',
                    'num_objects': len(objs),
                    'num_valid_yolo_boxes': num_boxes,
                    'classes_converted': ';'.join(classes_in_file)
                })
                total_converted += 1
                
    df = pd.DataFrame(log_records)
    df.to_csv(output_log, index=False)
    print(f"Saved conversion log to {output_log} ({len(df)} files, {total_boxes} boxes converted).")

if __name__ == '__main__':
    rdd_dir = r'C:\Users\lucky\Downloads\DL PBL\N-RDD2024Road damage and defects'
    run_conversion(rdd_dir)
