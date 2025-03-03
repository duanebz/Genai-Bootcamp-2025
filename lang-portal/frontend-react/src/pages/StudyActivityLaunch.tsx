import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useNavigation } from '@/context/NavigationContext'
import { createStudySession } from '../services/api'
import { Loader2 } from 'lucide-react'

const API_BASE_URL = 'http://localhost:5000'

interface StudyActivity {
  id: number
  title: string
  launch_url: string
  preview_url: string
}

interface Group {
  id: number
  name: string
}

interface LaunchData {
  activity: StudyActivity
  groups: Group[]
}

interface StudyActivityLaunchProps {
  onStart?: () => void
}

export default function StudyActivityLaunch({ onStart }: StudyActivityLaunchProps) {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { setCurrentStudyActivity } = useNavigation()
  const [launchData, setLaunchData] = useState<LaunchData | null>(null)
  const [selectedGroup, setSelectedGroup] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    const fetchLaunchData = async () => {
      if (!id) {
        setError(new Error('No activity ID provided'))
        setIsLoading(false)
        return
      }

      try {
        const response = await fetch(`${API_BASE_URL}/study-activities/${id}/launch`)
        if (!response.ok) {
          throw new Error(response.status === 404 ? 'Activity not found' : 'Failed to fetch launch data')
        }
        const data = await response.json()
        setLaunchData(data)
        setCurrentStudyActivity(data.activity)
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to load launch data'))
      } finally {
        setIsLoading(false)
      }
    }

    fetchLaunchData()
  }, [id, setCurrentStudyActivity])

  // Clean up when unmounting
  useEffect(() => {
    return () => {
      setCurrentStudyActivity(null)
    }
  }, [setCurrentStudyActivity])

  const handleLaunch = async () => {
    if (!launchData?.activity || !selectedGroup) return
    
    try {
      setIsLoading(true)
      // Create a study session first
      const result = await createStudySession(parseInt(selectedGroup), launchData.activity.id)
      const sessionId = result.session_id
      
      // Replace any instances of $group_id with the actual group id and add session_id
      const launchUrl = new URL(launchData.activity.launch_url)
      launchUrl.searchParams.set('group_id', selectedGroup)
      launchUrl.searchParams.set('session_id', sessionId.toString())
      
      // Call onStart if provided
      onStart?.()
      
      // Open the modified URL in a new tab
      window.open(launchUrl.toString(), '_blank')
      
      // Navigate to the session show page
      navigate(`/sessions/${sessionId}`)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to launch activity'))
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    )
  }

  if (error || !launchData) {
    return (
      <div className="text-center py-6">
        <p className="text-red-500 mb-4">
          {error?.message || 'Failed to load launch data'}
        </p>
        <Button
          variant="link"
          onClick={() => navigate('/activities')}
          className="text-blue-500 hover:text-blue-600"
        >
          Return to Activities
        </Button>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">{launchData.activity.title}</h1>
      
      <div className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Select Word Group</label>
          <Select onValueChange={setSelectedGroup} value={selectedGroup}>
            <SelectTrigger>
              <SelectValue placeholder="Select a word group" />
            </SelectTrigger>
            <SelectContent>
              {launchData.groups.map((group) => (
                <SelectItem key={group.id} value={group.id.toString()}>
                  {group.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button 
          onClick={handleLaunch}
          disabled={!selectedGroup || isLoading}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Launching...
            </>
          ) : (
            'Launch Now'
          )}
        </Button>
      </div>
    </div>
  )
}
