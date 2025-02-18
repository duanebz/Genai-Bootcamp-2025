import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import json
from app import create_app

class TestStudySessions(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock database cursor and IntegrityError
        self.mock_cursor = MagicMock()
        self.app.db = MagicMock()
        self.app.db.IntegrityError = Exception  # Mock the IntegrityError
        self.app.db.Error = Exception  # Mock the general DB Error
        self.app.db.cursor.return_value = self.mock_cursor

    def test_create_study_session_success(self):
        # Mock data
        test_data = {
            'group_id': 1,
            'study_activity_id': 1
        }
        self.mock_cursor.lastrowid = 1

        # Make request
        response = self.client.post(
            '/api/study-sessions',
            data=json.dumps(test_data),
            content_type='application/json'
        )

        # Assert response
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data['id'], 1)
        self.assertEqual(data['message'], 'Study session created successfully')

        # Verify the SQL query was called with correct parameters
        self.mock_cursor.execute.assert_called_once()
        call_args = self.mock_cursor.execute.call_args[0]
        self.assertIn('INSERT INTO study_sessions', call_args[0])

    def test_create_study_session_db_integrity_error(self):
        # Mock data
        test_data = {
            'group_id': 999,  # Non-existent ID
            'study_activity_id': 999
        }
        
        # Make cursor.execute raise IntegrityError
        self.mock_cursor.execute.side_effect = self.app.db.IntegrityError()

        # Make request
        response = self.client.post(
            '/api/study-sessions',
            data=json.dumps(test_data),
            content_type='application/json'
        )

        # Assert response
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('Invalid group_id or study_activity_id', data['error'])

    def test_create_study_session_missing_fields(self):
        # Test missing group_id
        response = self.client.post(
            '/api/study-sessions',
            data=json.dumps({'study_activity_id': 1}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('group_id is required', response.get_json()['error'])

        # Test missing study_activity_id
        response = self.client.post(
            '/api/study-sessions',
            data=json.dumps({'group_id': 1}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('study_activity_id is required', response.get_json()['error'])

        # Test missing data entirely
        response = self.client.post(
            '/api/study-sessions',
            data='',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('No data provided', response.get_json()['error'])

    def test_create_study_session_invalid_types(self):
        # Test invalid group_id type
        response = self.client.post(
            '/api/study-sessions',
            data=json.dumps({'group_id': '1', 'study_activity_id': 1}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('group_id must be an integer', response.get_json()['error'])

    def test_get_study_sessions(self):
        # Mock database response
        mock_sessions = [{
            'id': 1,
            'group_id': 1,
            'group_name': 'Test Group',
            'activity_id': 1,
            'activity_name': 'Test Activity',
            'created_at': datetime.now().isoformat(),
            'review_items_count': 5
        }]
        self.mock_cursor.fetchall.return_value = mock_sessions
        self.mock_cursor.fetchone.return_value = {'count': 1}

        # Make request with pagination parameters
        response = self.client.get('/api/study-sessions?page=1&per_page=10')

        # Assert response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('items', data)
        self.assertIn('total', data)
        self.assertEqual(data['page'], 1)
        self.assertEqual(data['per_page'], 10)
        self.assertEqual(data['total_pages'], 1)

        # Verify SQL queries were called
        self.assertEqual(self.mock_cursor.execute.call_count, 2)  # One for count, one for data

    def test_get_study_session_by_id(self):
        # Mock database response
        mock_session = {
            'id': 1,
            'group_id': 1,
            'group_name': 'Test Group',
            'activity_id': 1,
            'activity_name': 'Test Activity',
            'created_at': datetime.now().isoformat(),
            'review_items_count': 5
        }
        mock_words = [{
            'id': 1,
            'kanji': '漢字',
            'romaji': 'kanji',
            'english': 'chinese characters',
            'session_correct_count': 3,
            'session_wrong_count': 1
        }]
        self.mock_cursor.fetchone.side_effect = [mock_session, {'count': 1}]
        self.mock_cursor.fetchall.return_value = mock_words

        # Make request
        response = self.client.get('/api/study-sessions/1')

        # Assert response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('session', data)
        self.assertIn('words', data)
        self.assertEqual(len(data['words']), 1)
        self.assertEqual(data['words'][0]['kanji'], '漢字')

    def test_get_nonexistent_study_session(self):
        # Mock database response for non-existent session
        self.mock_cursor.fetchone.return_value = None

        # Make request
        response = self.client.get('/api/study-sessions/999')

        # Assert response
        self.assertEqual(response.status_code, 404)
        self.assertIn('error', json.loads(response.data))

    def test_reset_study_sessions(self):
        # Make request
        response = self.client.post('/api/study-sessions/reset')

        # Assert response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Study history cleared successfully')

        # Verify database calls in correct order
        expected_calls = [
            unittest.mock.call('DELETE FROM word_review_items'),
            unittest.mock.call('DELETE FROM study_sessions')
        ]
        self.mock_cursor.execute.assert_has_calls(expected_calls, any_order=False)
        self.app.db.commit.assert_called_once()

    def test_reset_study_sessions_error(self):
        # Make cursor.execute raise an error
        self.mock_cursor.execute.side_effect = Exception('Database error')

        # Make request
        response = self.client.post('/api/study-sessions/reset')

        # Assert response
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main() 