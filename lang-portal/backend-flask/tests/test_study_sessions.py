import unittest
import json
from app import create_app

class TestStudySessions(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.testing = True
        self.client = self.app.test_client()
        
        # Reset study sessions before each test
        self.client.post('/api/study-sessions/reset')

    def test_create_study_session_success(self):
        # Test successful creation
        response = self.client.post('/api/study-sessions',
            data=json.dumps({
                'group_id': 1,
                'study_activity_id': 1
            }),
            content_type='application/json'
        )
        
        data = response.get_json()
        self.assertEqual(response.status_code, 201)
        self.assertIn('id', data)
        self.assertIn('message', data)
        self.assertTrue(isinstance(data['id'], int))

    def test_create_study_session_missing_fields(self):
        # Test missing group_id
        response = self.client.post('/api/study-sessions',
            data=json.dumps({
                'study_activity_id': 1
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()['error'],
            'group_id is required'
        )

        # Test missing study_activity_id
        response = self.client.post('/api/study-sessions',
            data=json.dumps({
                'group_id': 1
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()['error'],
            'study_activity_id is required'
        )

    def test_create_study_session_invalid_data(self):
        # Test invalid group_id type
        response = self.client.post('/api/study-sessions',
            data=json.dumps({
                'group_id': "1",  # string instead of int
                'study_activity_id': 1
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()['error'],
            'group_id must be an integer'
        )

        # Test non-existent group_id
        response = self.client.post('/api/study-sessions',
            data=json.dumps({
                'group_id': 99999,  # non-existent id
                'study_activity_id': 1
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid group_id', response.get_json()['error'])

if __name__ == '__main__':
    unittest.main() 