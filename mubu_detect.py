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

def getmubu(video_path, mubu_point):
    '''
    :param video_path:  视频存放路径
    :param mubu_point:  幕布位置，（左上，右下）
    :return:
        begin_point_h, 开始收起幕布时，确认的幕布下边沿相对高度
        end_point_h, 幕布不再移动时，确认的幕布下边沿相对高度
        begin_image, 初始时刻截图
        frame, 结束时刻截图
    '''
    point_lt, point_rb = mubu_point
    begin_work = False
    begin_point_h = 0
    end_point_h = 0
    pass_time = 0
    is_end = False

    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()

    if ret:
        frame = image[point_lt[1]:point_rb[1], point_lt[0]:point_rb[0]]
        cv2.imshow('f',frame)
        cv2.waitKey(0)
        begin_image = frame.copy()
        h, w, c = frame.shape
        point_h = int(h * 0.1)
        count = 0
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while ret:
            count += 1
            ret, image = cap.read()
            frame = image[point_lt[1]:point_rb[1], point_lt[0]:point_rb[0]]
            frame_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            fgmask = fgbg.apply(frame_gray)


            number_change = len(fgmask[fgmask > 0])
            ####### 幕布开始上升 且 连续5帧基本无变化，视为终止条件1 #####
            if (number_change < w * h * 0.001) & (begin_work):
                pass_time += 1
                if pass_time>5:
                    return begin_point_h, end_point_h, begin_image, frame, begin_image_real


            if (number_change > w * h / 3):# | (number_change < w * h * 0.001):
                # print('this frame change much which is pass :{}'.format(count))
                continue

            ###### 寻找幕布开始上升时下边沿位置 ######
            if not begin_work:
                _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_contours = 0
                fgmask_tmp = fgmask.copy()*0
                for cont in contours:
                    area_tmp = cv2.contourArea(cont)
                    if area_tmp < 30:
                        continue
                    num_contours += 1
                    number, _, _ = cont.shape
                    for i in range(number):
                        fgmask_tmp[cont[i, 0, 1], cont[i, 0, 0]] = 255

                if num_contours>10:
                    # print('this frame has too many noisy which is pass :{}'.format(count))
                    continue

                count_pixels_in_cols = 0
                total_h=0
                num_h=0
                fgmask_tmp[:int(0.3*h),:]=0
                for i in range(fgmask_tmp.shape[1]):
                    a = np.where(fgmask_tmp[:, i] > 0)
                    if len(a[0]):
                        count_pixels_in_cols += 1
                        total_h += sum(a[0])
                        num_h += len(a[0])
                if count_pixels_in_cols<(w/2):
                    # print('width of this frame change less than half of whole R which is pass :{}'.format(count))
                    continue
                end_point_h = int(total_h/num_h)
                cv2.imshow('fgmt',fgmask_tmp)

                total_h_2 = 0
                num_h_2 = 0
                fgmask_tmp[:int(0.7 * h), :] = 0
                for i in range(fgmask_tmp.shape[1]):
                    a = np.where(fgmask_tmp[:, i] > 0)
                    if len(a[0]):
                        total_h_2 += sum(a[0])
                        num_h_2 += len(a[0])

                if num_h_2>num_h/3:
                    end_point_h = int(total_h / num_h)
                begin_point_h = end_point_h

            fgmask_cut = fgmask[max(0,end_point_h-point_h):min(h,end_point_h+point_h),:]

            # ###### 所用到的变化框在原图上的位置可视化  #########
            cv2.rectangle(frame,(0,max(0,end_point_h-point_h)),(w,min(h,end_point_h+point_h)),color=(0,255,0))
            cv2.imshow('frame',frame)

            _, contours, hierarchy = cv2.findContours(fgmask_cut.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_total=[]
            Area=0
            for cont in contours:
                area_tmp=cv2.contourArea(cont)
                if area_tmp<30:
                    continue
                Area += cv2.contourArea(cont)
                if len(contours_total):
                    contours_total = np.concatenate((contours_total, cont), axis=0)
                else:
                    contours_total = cont.copy()
            if Area>30:
                if not begin_work:
                    begin_point_h = end_point_h
                    begin_image_real = frame
                    begin_work = True
                cv2.imshow('fgm',fgmask)
                cv2.imshow('fra',frame)
                # cv2.waitKey(0)

                ##### 幕布下边沿直线可视化 #####
                rows,cols=frame.shape[:2]
                [vx,vy,x,y]=cv2.fitLine(contours_total,cv2.DIST_L2,0,0.01,0.01)
                lefty=int((-x*vy/vx)+y)
                righty=int(((cols-x)*vy/vx)+y)
                cv2.line(frame,(cols-1,righty+max(0,end_point_h-point_h)),(0,lefty+max(0,end_point_h-point_h)),(255, 0, 0),2)
                cv2.imshow('frame_cut',frame)
                cX, cY = my_meanpoint(contours_total,fgmask_cut)
                cY += int(max(0, end_point_h - point_h))
                end_point_h = cY

                if begin_point_h<end_point_h:
                    begin_point_h=end_point_h
                    begin_image_real = frame
                ######### 幕布下边沿质心可视化 #######
                cv2.circle(frame, (cX, cY), 15, (0, 0, 255), -1)
                cv2.imshow('frame',frame)


                if not is_end:
                    if cY < 0.1 * h:
                        is_end = True

            ###### 幕布下边沿到达指定区域 且 当前帧无变化，视为终止条件2 #######
            if is_end & (len(contours_total)==0):
                # print('everything is ok !')
                # print(count)
                # is_end=False
                # cv2.waitKey(0)
                return begin_point_h, end_point_h, begin_image, frame, begin_image_real

            #########  中途中断设置 ##########
            # k = cv2.waitKey(10) & 0xff  # 按esc退出
            # if k == 27:
            #     break
        cap.release()
        cv2.destoryAllWindows()  # 关闭所有窗口
    else:
        print('error video !')


if __name__ == "__main__":
    video_path='./13F-8.avi'
    mubu_point=((868, 438), (1164, 634))
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    mubu_point = get_rect(image, title='get_rect')  # 手动选框
    print(mubu_point)
    begin_point_h, end_point_h, begin_image, end_image, begin_image_real = getmubu(video_path=video_path, mubu_point=mubu_point)

    threshold = (mubu_point[1][1] - mubu_point[0][1]) * 0.1
    if abs(begin_point_h-(mubu_point[1][1] - mubu_point[0][1]))<threshold:
        print('成功放下')
    else:
        print('初始位置异常!')

    if end_point_h<threshold:
        print('成功收起')
    else:
        print('最终位置异常!')

    print(begin_point_h, end_point_h)
    cv2.imshow('first-frame', begin_image)
    cv2.imshow('begin-frame', begin_image_real)
    cv2.imshow('end-frame', end_image)
    cv2.waitKey(0)
