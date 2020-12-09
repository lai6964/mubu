import numpy as np
import cv2

def my_meanpoint(contours,fgmask_cut):
    '''
    :param contours: 多个变化区域轮廓边缘点集合
    :param fgmask_cut: 截取区域的图像
    :return: 质心
    '''
    number, _, _ = contours.shape
    sum_x=sum_y=0
    frame_cut_tmp = fgmask_cut.copy() / 255#127
    for i in range(number):
        sum_x += contours[i,0,0] * (frame_cut_tmp[contours[i,0,1],contours[i,0,0]]+1)
        sum_y += contours[i,0,1] * (frame_cut_tmp[contours[i,0,1],contours[i,0,0]]+1)
        number += int(frame_cut_tmp[contours[i,0,1],contours[i,0,0]])
    mean_x = sum_x/number
    mean_y = sum_y/number
    return int(mean_x), int(mean_y)

def get_rect(im, title='get_rect'):
    mouse_params = {'tl': None, 'br': None, 'current_pos': None, 'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                          mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
          min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
          max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)


def mubu_detect(fgmask, begin_work, begin_point_h, end_point_h, is_end, point_h, pass_time):
    h, w = fgmask.shape

    number_change = len(fgmask[fgmask > 0])
    ####### 幕布开始上升 且 连续5帧基本无变化，视为终止条件1 #####

    if (number_change < w*h*0.001) & (begin_work):
        pass_time += 1
        if pass_time > 5:
            return True, begin_work, begin_point_h, end_point_h
    if (number_change > image.shape[0] * image.shape[1] / 3):  # | (number_change < w * h * 0.001):
        # print('this frame change much which is pass :{}'.format(count))
        return False, begin_work, begin_point_h, end_point_h

    if not begin_work:
        _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = 0
        fgmask_tmp = fgmask.copy() * 0
        for cont in contours:
            area_tmp = cv2.contourArea(cont)
            if area_tmp < 30:
                continue
            num_contours += 1
            number, _, _ = cont.shape
            for i in range(number):
                fgmask_tmp[cont[i, 0, 1], cont[i, 0, 0]] = 255

        if num_contours > 10:
            # print('this frame has too many noisy which is pass :{}'.format(count))
            return is_end, begin_work, begin_point_h, end_point_h

        count_pixels_in_cols = 0
        total_h = 0
        num_h = 0
        fgmask_tmp[:int(0.3 * h), :] = 0
        for i in range(fgmask_tmp.shape[1]):
            a = np.where(fgmask_tmp[:, i] > 0)
            if len(a[0]):
                count_pixels_in_cols += 1
                total_h += sum(a[0])
                num_h += len(a[0])
        if count_pixels_in_cols < (w / 2):
            # print('width of this frame change less than half of whole R which is pass :{}'.format(count))
            return is_end, begin_work, begin_point_h, end_point_h
        end_point_h = int(total_h / num_h)
        # cv2.imshow('fgmt',fgmask_tmp)

        total_h_2 = 0
        num_h_2 = 0
        fgmask_tmp[:int(0.7 * h), :] = 0
        for i in range(fgmask_tmp.shape[1]):
            a = np.where(fgmask_tmp[:, i] > 0)
            if len(a[0]):
                total_h_2 += sum(a[0])
                num_h_2 += len(a[0])

        if num_h_2 > num_h / 3:
            end_point_h = int(total_h / num_h)
        begin_point_h = end_point_h

    fgmask_cut = fgmask[max(0, end_point_h - point_h):min(h, end_point_h + point_h), :]

    # ###### 所用到的变化框在原图上的位置可视化  #########
    cv2.rectangle(image, (0, max(0, end_point_h - point_h)), (w, min(h, end_point_h + point_h)), color=(255, 0, 0))
    # cv2.imshow('frame',frame)

    _, contours, hierarchy = cv2.findContours(fgmask_cut.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_total = []
    Area = 0
    for cont in contours:
        area_tmp = cv2.contourArea(cont)
        if area_tmp < 30:
            continue
        Area += cv2.contourArea(cont)
        if len(contours_total):
            contours_total = np.concatenate((contours_total, cont), axis=0)
        else:
            contours_total = cont.copy()
    if Area > 30:
        if not begin_work:
            begin_point_h = end_point_h
            begin_image_real = image
            begin_work = True
        # cv2.imshow('fgm',fgmask)
        # cv2.imshow('fra',frame)
        # cv2.waitKey(0)

        ##### 幕布下边沿直线可视化 #####
        rows,cols=fgmask_cut.shape[:2]
        [vx,vy,x,y]=cv2.fitLine(contours_total,cv2.DIST_L2,0,0.01,0.01)
        lefty=int((-x*vy/vx)+y)
        righty=int(((cols-x)*vy/vx)+y)
        cv2.line(fgmask_cut,(cols-1,righty),(0,lefty),(255, 0, 0),2)
        cv2.imshow('frame_cut',fgmask_cut)
        cX, cY = my_meanpoint(contours_total, fgmask_cut)
        cY += int(max(0, end_point_h - point_h))
        end_point_h = cY

        if begin_point_h < end_point_h:
            begin_point_h = end_point_h
            begin_image_real = image

        ######### 幕布下边沿质心可视化 #######
        cv2.circle(fgmask_cut, (cX, cY), 15, (0, 0, 255), -1)
        cv2.imshow('frame',fgmask)
        cv2.waitKey(100)

        if not is_end:
            if cY < 0.1 * h:
                is_end = True

    ###### 幕布下边沿到达指定区域 且 当前帧无变化，视为终止条件2 #######
    if is_end & (len(contours_total) == 0):
        # print('everything is ok !')
        # print(count)
        # is_end=False
        # cv2.waitKey(0)
        return is_end, begin_work, begin_point_h, end_point_h
    else:
        return is_end, begin_work, begin_point_h, end_point_h


def getmubu(video_path, point_list):
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    fgbg = cv2.createBackgroundSubtractorMOG2()
    if ret:
        point_lt=[]
        point_rb=[]
        begin_work = []
        begin_point_h = []
        end_point_h = []
        pass_time = []
        is_end = []
        point_h= []
        count=0
        frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for mubu_point in point_list:
            point_lt.append(mubu_point[0])
            point_rb.append(mubu_point[1])

            pass_time.append(0)
            begin_work.append(False)
            begin_point_h.append(0)
            end_point_h.append(0)
            is_end.append(False)
            point_h.append(int((mubu_point[1][1]-mubu_point[0][1])*0.1))

        while ret:
            # print(count)
            # count += 1
            pic_before = frame_gray.copy()
            frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fgmask = fgbg.apply(frame_gray)

            diff = cv2.absdiff(frame_gray, pic_before)
            retVal, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

            sort_frame = thresh.copy()
            sort_frame[fgmask==0]=0

            is_all_end = True
            for i in range(len(point_list)):
                image = sort_frame[point_lt[i][1]:point_rb[i][1], point_lt[i][0]:point_rb[i][0]]
                is_end[i], begin_work[i], begin_point_h[i], end_point_h[i] = mubu_detect(image,
                                                                                      begin_work[i],
                                                                                      begin_point_h[i],
                                                                                      end_point_h[i],
                                                                                      is_end[i],
                                                                                      point_h[i],
                                                                                      pass_time[i])
                is_all_end = is_all_end & is_end[i]
            if is_all_end:
                return begin_point_h, end_point_h


            ret, image = cap.read()
        return begin_point_h, end_point_h

    else:
        print('error video path')


if __name__ == "__main__":
    video_path='./2020-11-26-11-28-34-462.avi'
    # mubu_point=((868, 438), (1164, 634))
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    mubu_list=[]

    #设定手选框个数
    for i in range(1):
        mubu_point = get_rect(image, title='get_rect')  # 手动选框
        print(mubu_point)
        mubu_list.append(mubu_point)

    begin_point_h, end_point_h = getmubu(video_path=video_path, point_list=mubu_list)

    print(begin_point_h,end_point_h)

    for i in range(len(mubu_list)):
        print(i)

        if (begin_point_h[i]==0) & (end_point_h[i]==0):
            print('未检测到任何幕布移动！')
        else:

            threshold = (mubu_list[i][1][1] - mubu_list[i][0][1]) * 0.1
            if abs(begin_point_h[i]-(mubu_list[i][1][1] - mubu_list[i][0][1]))<threshold:
                print('成功放下')
            else:
                print('初始位置异常!')

            if end_point_h[i]<threshold:
                print('成功收起')
            else:
                print('最终位置异常!')


