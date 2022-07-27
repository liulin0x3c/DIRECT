import math

import cv2
import numpy as np


def fun_analysis(x):
    '''
    here you can design your f(x).in this case, the function is sigma（xi-1/3）²
    :param x:[x0,x1,x2,x3,x4,x5......]
    :return:the value of f(x) (float type)
    '''
    y = 0.
    for i in x:
        y += pow(i - 1 / 3, 2)
    return float(y)


def DIRECT(f, bands, iterator_num):
    '''
    DIRECT implementation.
    :param f: the f(x) you need analysis. such as fun_analysis(x).
    :param bands:
    :param iterator_num:
    :return: min
    '''
    l = get_len_of_interval(bands[0]) / 2
    dimensions = len(bands)

    center_point = get_center_point(bands)

    convert_bands = np.zeros((dimensions, 2), dtype=np.longdouble)
    for dimension in range(dimensions):
        convert_bands[dimension] = [0, 1]

    def fun_convert(x):
        convert_x = []
        for i in range(dimensions):
            convert_x.append(x[i] * 2 * l + center_point[i] - l)
        return f(convert_x)
    return half_DIRECT(fun_convert, convert_bands, iterator_num)


def get_center_point(bands):
    """
    返回搜索区域的中心点
    :param bands:搜索区域
    :return:中心点
    """
    dimensions = len(bands)
    center_point = np.empty((dimensions, 1))
    for i in range(dimensions):
        center_point[i] = (bands[i][0] + bands[i][1]) / 2
    return center_point


def get_len_of_interval(interval):
    return interval[1] - interval[0]


def divde_one_to_tree_block(bands, need_divided_dimension):
    delta = get_len_of_interval(bands[need_divided_dimension]) / 3
    divided_bands_list = []
    divided_bands = bands.copy()
    divided_bands[need_divided_dimension][1] -= delta * 2
    divided_bands_list.append(divided_bands)

    divided_bands = bands.copy()
    divided_bands[need_divided_dimension][0] += delta
    divided_bands[need_divided_dimension][1] -= delta
    divided_bands_list.append(divided_bands)

    divided_bands = bands.copy()
    divided_bands[need_divided_dimension][0] += delta * 2
    divided_bands_list.append(divided_bands)

    return divided_bands_list


def get_max_side_dimension_list(bands):
    dimensions = len(bands)
    max_side_length = np.longdouble(0)
    # aka I
    max_side_dimension_list = []
    for dimension in range(dimensions):
        length = get_len_of_interval(bands[dimension])
        if math.isclose(max_side_length, length):
            max_side_dimension_list.append(dimension)
        elif max_side_length < length:
            max_side_dimension_list.clear()
            max_side_length = length
            max_side_dimension_list.append(dimension)
    return max_side_dimension_list


def remove_array(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def get_divided_bands_list(f, bands):
    max_side_dimension_list = get_max_side_dimension_list(bands)

    # aka W
    # 找到要切割的维度顺序
    delta = get_len_of_interval(bands[max_side_dimension_list[0]]) / 3
    center_point = get_center_point(bands)

    def get_wi(dimension):
        """
        :param dimension: 某一维度
        :return: wi
        """
        a_point = center_point.copy()
        a_point[dimension] -= delta
        b_point = center_point.copy()
        b_point[dimension] += delta
        fa = f(a_point)
        fb = f(b_point)
        wi = min(fa, fb)
        return wi

    get_wi(max_side_dimension_list[0])
    sorted_max_side_dimension_list = sorted(max_side_dimension_list, key=get_wi)

    # 切割出三个搜索区域
    divided_bands_list = [bands]
    need_divide_bands = bands
    for dimension in sorted_max_side_dimension_list:
        remove_array(divided_bands_list, need_divide_bands)
        divided_bands_list += divde_one_to_tree_block(need_divide_bands, dimension)
        need_divide_bands = divided_bands_list[-2]
    return divided_bands_list


def sort_points(point_array):
    """Return point_array sorted by lftmost first, then by slopee, ascending."""
    def slope(y):
        """returns the slope of the 2 points."""
        x = point_array[0]
        if math.isclose(y[0], x[0]):
            return x[1] - y[1] / 0.00000000001

        return (x[1] - y[1]) / \
               (x[0] - y[0])

    def k(point):
        return point[0]

    point_array.sort(key=k)  # put leftmost first
    point_array = point_array[:1] + sorted(point_array[1:], key=slope)
    return point_array


def graham_scan(point_array):
    """Takes an array of points to be scanned.
    Returns an array of points that make up the convex hull surrounding the points passed in in point_array.
    """
    def cross_product_orientation(a, b, c):
        """Returns the orientation of the set of points.
        >0 if x,y,z are clockwise, <0 if counterclockwise, 0 if co-linear.
        """
        return (b[1] - a[1]) * \
               (c[0] - a[0]) - \
               (b[0] - a[0]) * \
               (c[1] - a[1])

    # convex_hull is a stack of points beginning with the leftmost point.
    convex_hull = []
    sorted_points = sort_points(point_array)
    for p in sorted_points:
        # if we turn clockwise to reach this point, pop the last point from the stack, else, append this point to it.
        while len(convex_hull) > 1 and cross_product_orientation(convex_hull[-2], convex_hull[-1], p) >= 0:
            convex_hull.pop()
        convex_hull.append(p)
    # the stack is now a representation of the convex hull, return it.
    return convex_hull


def select_bands(f, bands_list):
    points = []
    for bands in bands_list:
        center_point = get_center_point(bands)
        dis = 0.
        dimensions = len(bands)
        for i in range(dimensions):
            dis += math.pow(bands[i][0] - center_point[i], 2)
        dis = math.sqrt(dis)
        point = [dis, f(center_point), bands.copy()]
        points.append(point)
    hull = graham_scan(points)
    selected_bands = []
    for point in hull:
        selected_bands.append(point[2])
    return selected_bands


def half_DIRECT(f, bands, iterator_num):
    """
    针对单位超立方体的搜索范围内的DIRECT算法实现
    :param f:函数f(x)
    :param bands:单位超立方体的搜索范围
    :param iterator_num:迭代次数
    :return:
    """
    # 为min_f设置初始值
    center_point = get_center_point(bands)
    min_f = f(center_point)

    # 初始状态，bands是唯一的超立方体
    total_bands_list = [bands]
    pure_divided_bands_list = [bands]
    for i in range(iterator_num):
        bands_list = select_bands(f, pure_divided_bands_list)
        pure_divided_bands_list = []
        for bands in bands_list:
            remove_array(total_bands_list, bands)
            pure_divided_bands_list += get_divided_bands_list(fun_analysis, bands)
            for divided_bands in pure_divided_bands_list:
                center_point = get_center_point(divided_bands)
                min_f = min(f(center_point), min_f)
            total_bands_list += pure_divided_bands_list
        # print("iterator: ", end='')
        # print(i, end='')
        # print("\tmin:", end='')
        # print(min_f)
    return min_f


# def draw_points(img, points_list):
#     point_size = 1
#     point_color = (0, 0, 255)  # BGR
#     thickness = 4  # 可以为 0 、4、8
#     for point in points_list:
#         p = [int(point[0] * 500) + 10, int(point[1] * 500) + 10]
#         cv2.circle(img, p, point_size, point_color, thickness)


# def draw_line(img, points_list):
#     point_color = (0, 255, 0)  # BGR
#     thickness = 1
#     lineType = 4
#     for i in range(len(points_list) - 1):
#         point1 = points_list[i]
#         p1 = [int(point1[0] * 500) + 10, int(point1[1] * 500) + 10]
#         point2 = points_list[i + 1]
#         p2 = [int(point2[0] * 500) + 10, int(point2[1] * 500) + 10]
#         cv2.line(img, p1, p2, point_color, thickness, lineType)


# def draw_2D_define(img, bands_2D):
#     bands = bands_2D
#     x_low = bands[0][0]
#     x_high = bands[0][1]
#     y_low = bands[1][0]
#     y_high = bands[1][1]
#     draw_line(img, [[x_low, y_low], [x_low, y_high], [x_high, y_high], [x_high, y_low], [x_low, y_low]])
#

if __name__ == '__main__':
    # img = np.zeros((520, 520, 3), np.uint8)
    iterator_num = 10
    dimensions = 2
    bands = np.zeros((dimensions, 2), dtype=np.longdouble)
    for i in range(dimensions):
        bands[i] = [0, 1]

    min_f = DIRECT(fun_analysis, bands, iterator_num)
    print(min_f)
