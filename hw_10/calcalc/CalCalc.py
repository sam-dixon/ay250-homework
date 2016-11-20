"""
Interface for Wolfram Alpha
"""

import argparse
import xml.etree.ElementTree as et
import requests

APPID = 'J5TQ9X-6EGKUG8WT2'

def parse():
    """Parse command line options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help='Query string to send to WA',
                        dest='q')
    parser.add_argument('-f', help='Get the answer as a float (no units)',
                        dest='flt', action='store_true')
    return parser.parse_args()


def calculate(query, return_float=False):
    """Evaluates the input string with WA API"""
    params = {'appid': APPID,
              'format': 'plaintext',
              'input': query}
    result = requests.get('http://api.wolframalpha.com/v2/query',
                          params=params)
    tree = et.fromstring(result.content)
    if tree.get('success') == 'false':
        return "Wolfram Alpha didn't understand your query"
    for pod in tree.iter('pod'):
        if pod.get('title') == 'Result':
            result_text = pod[0][0].text
            if not return_float:
                return result_text
            else:
                return text2float(result_text)
    return "Wolfram Alpha didn't understand your query"


def text2float(text):
    """Convert text with units to a float"""
    if text[0].isdigit():
        remove_unit = text.split()[0]
        if b'\xc3\x97' in remove_unit.encode():
            prefix, power = [x.decode() for x in remove_unit.encode().split(b'\xc3\x97')]
            base, exp = power.split('^')
            return float(prefix)*float(base)**float(exp)
        return float(remove_unit)
    print("Can't convert to float")
    return text


def test_simple():
    assert calculate('3333*3333') == '11108889'


def test_no_numerical_result():
    fail_msg = "Wolfram Alpha didn't understand your query"
    assert calculate('berkeley') == fail_msg


def test_success_false():
    fail_msg = "Wolfram Alpha didn't understand your query"
    assert calculate('sssssss') == fail_msg


def test_simple_text2float():
    assert calculate('3333*3333', return_float=True) == 11108889.


def test_text2float_with_unit():
    assert text2float('0.4536 kg  (kilograms)') == 0.4536


def test_text2float_fail():
    hgf_float = calculate('highest grossing film', return_float=True)
    hgf_str = calculate('highest grossing film', return_float=False)
    assert hgf_float == hgf_str


def test_sci_notation():
    assert text2float('1.7×10^3') == 1700


def test_sci_notation_with_unit():
    solar_mass_result = text2float('5.965305×10^31 kg  (kilograms)')
    float_value = 5.965305e31
    assert solar_mass_result == float_value


if __name__ == '__main__':
    args = parse()
    if args.q is not None:
        print(calculate(args.q, return_float=args.flt))
