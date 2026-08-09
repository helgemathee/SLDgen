import { useEffect, useState } from 'react'

/**
 * A hash router in thirty lines (Spec 3 SS2).
 *
 * Four routes and no nested layouts, so a routing library would be weight
 * without benefit. The hash also means the API can serve the SPA from a plain
 * static mount: the server never sees a client route, so there is no catch-all
 * rewrite to get wrong.
 */

export type Route =
  | { name: 'jobs' }
  | { name: 'job'; id: string }
  | { name: 'compare'; ids: string[] }
  | { name: 'new' }

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#/, '') || '/jobs'
  const [pathname, query] = path.split('?')
  const parts = pathname.split('/').filter(Boolean)

  if (parts[0] === 'jobs' && parts[1]) return { name: 'job', id: parts[1] }
  if (parts[0] === 'compare') {
    const ids = new URLSearchParams(query ?? '').get('ids')
    return { name: 'compare', ids: ids ? ids.split(',').filter(Boolean) : [] }
  }
  if (parts[0] === 'new') return { name: 'new' }
  return { name: 'jobs' }
}

export function routeHref(route: Route): string {
  switch (route.name) {
    case 'job':
      return `#/jobs/${route.id}`
    case 'compare':
      return `#/compare?ids=${route.ids.join(',')}`
    case 'new':
      return '#/new'
    default:
      return '#/jobs'
  }
}

export function navigate(route: Route) {
  window.location.hash = routeHref(route)
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash))
  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  return route
}
