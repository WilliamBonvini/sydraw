from datafact import modelgen
def test_haversine():
    assert modelgen.haversine(52.370216, 4.895168, 52.520008,
                              13.404954) == 945793.4375088713


def test_generate_circle():
    outliers_rate_range = [0.10]
    dirs = ['fakedata', '', 'test']
    modelgen.generate_data(num_samples=1,
                     num_points_per_sample=2,
                     outliers_rate_range=outliers_rate_range,
                     model='line',
                     noise_perc=0.01,
                     dest='matlab',
                     dirs=dirs)



